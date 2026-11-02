// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Two-host GPUDirect transport. CUDA stream memory operations drive a CPU
// verbs proxy; no communication kernel competes with native MegaMoE's grid.
#include <cuda.h>
#include <infiniband/verbs.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
namespace {
constexpr size_t kPage = 65536;
constexpr int kReceives = 512;
constexpr uint64_t kControlSend = uint64_t{1} << 63;

void cuda_check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  throw std::runtime_error(std::string(operation) + ": " +
                           (message ? message : "unknown CUDA error"));
}
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
uint64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
uint32_t load(const uint32_t* address) {
  return __atomic_load_n(address, __ATOMIC_ACQUIRE);
}
void store(uint32_t* address, uint32_t value) {
  __atomic_store_n(address, value, __ATOMIC_RELEASE);
}

struct Region {
  ibv_mr* mr = nullptr;
  int fd = -1;
  Region(ibv_pd* pd, uint64_t address, size_t bytes) {
    require(address % kPage == 0 && bytes % kPage == 0,
            "RDMA buffers must be 64-KiB aligned and padded");
    unsigned int sync = 1;
    cuda_check(
        cuPointerSetAttribute(&sync, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, address),
        "enable synchronous memory operations");
    cuda_check(cuMemGetHandleForAddressRange(
                   &fd, address, bytes, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0),
               "export GPU DMA-BUF");
    mr = ibv_reg_dmabuf_mr(pd, 0, bytes, address, fd,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
      ::close(fd);
      fd = -1;
      throw std::runtime_error("ibv_reg_dmabuf_mr: " +
                               std::string(std::strerror(errno)));
    }
  }
  ~Region() {
    if (mr) ibv_dereg_mr(mr);
    if (fd >= 0) ::close(fd);
  }
};

struct alignas(64) Flags {
  alignas(64) uint32_t request;
  alignas(64) uint32_t done;
};
struct Credit {
  uint64_t epoch;
  uint32_t job;
  uint32_t reserved;
};
struct Trace {
  uint64_t epoch, request_ns, post_ns, complete_ns, bytes;
  uint32_t job;
};
struct Job {
  std::unique_ptr<Region> send_region, receive_region;
  uint64_t send, receive, remote_receive = 0;
  uint32_t remote_key = 0;
  size_t bytes;
  Flags* flags = nullptr;
  CUdeviceptr device_flags = 0;
  bool connected = false, active = false, posted = false;
  uint64_t epoch = 0, peer_epoch = 0, request_ns = 0, post_ns = 0;
  uint32_t sent_mask = 0, received_mask = 0;
  ~Job() {
    if (flags) cuMemFreeHost(flags);
  }
};

class Endpoint {
 public:
  Endpoint(const std::string& hca, int port, int gid_index, int qps,
           int traffic_class, int timeout_seconds, int self)
      : port_(port),
        gid_index_(gid_index),
        count_(qps),
        traffic_class_(traffic_class),
        self_(self),
        timeout_ns_(uint64_t(timeout_seconds) * 1000000000) {
    require(qps >= 1 && qps <= 8 && (self == 0 || self == 1),
            "Expected 1..8 QPs and a two-host lane rank");
    require(port >= 1 && port <= 255 && gid_index >= 0 && gid_index <= 255 &&
                traffic_class >= 0 && traffic_class <= 255 &&
                timeout_seconds > 0,
            "Invalid RDMA configuration");
    try {
      cuda_check(cuCtxGetCurrent(&cuda_context_), "get current CUDA context");
      require(cuda_context_ != nullptr, "Initialize CUDA before RDMA");
      CUdevice device;
      cuda_check(cuCtxGetDevice(&device), "get CUDA device");
      int ordering = 0, flush = 0;
      cuda_check(
          cuDeviceGetAttribute(
              &ordering, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
              device),
          "query GPUDirect ordering");
      cuda_check(
          cuDeviceGetAttribute(
              &flush, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
              device),
          "query GPUDirect flush support");
      flush_ = ordering < CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER;
      require(!flush_ || (flush & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST),
              "GPU cannot make RDMA receives visible to the compute context");
      int devices = 0;
      ibv_device** list = ibv_get_device_list(&devices);
      require(list != nullptr, "Cannot enumerate RDMA devices");
      for (int i = 0; i < devices; ++i)
        if (hca == ibv_get_device_name(list[i]))
          context_ = ibv_open_device(list[i]);
      ibv_free_device_list(list);
      require(context_ != nullptr, "Cannot open RDMA device " + hca);
      require(ibv_query_port(context_, port_, &port_attr_) == 0,
              "Cannot query RDMA port");
      require(port_attr_.state == IBV_PORT_ACTIVE &&
                  port_attr_.link_layer == IBV_LINK_LAYER_ETHERNET,
              "Expected an active RoCE port");
      require(ibv_query_gid(context_, port_, gid_index_, &gid_) == 0,
              "Cannot query RoCE GID");
      pd_ = ibv_alloc_pd(context_);
      require(pd_ != nullptr, "Cannot allocate protection domain");
      cq_ =
          ibv_create_cq(context_, count_ * kReceives * 2, nullptr, nullptr, 0);
      require(cq_ != nullptr, "Cannot create completion queue");
      credits_.resize(count_ * kReceives);
      credit_mr_ =
          ibv_reg_mr(pd_, credits_.data(), credits_.size() * sizeof(Credit),
                     IBV_ACCESS_LOCAL_WRITE);
      require(credit_mr_ != nullptr, "Cannot register control receive memory");
      std::random_device random;
      for (int q = 0; q < count_; ++q) {
        ibv_qp_init_attr init{};
        init.send_cq = init.recv_cq = cq_;
        init.qp_type = IBV_QPT_RC;
        init.cap.max_send_wr = kReceives;
        init.cap.max_recv_wr = kReceives;
        init.cap.max_send_sge = init.cap.max_recv_sge = 1;
        init.cap.max_inline_data = sizeof(Credit);
        ibv_qp* qp = ibv_create_qp(pd_, &init);
        require(qp != nullptr, "Cannot create RC queue pair");
        qps_.push_back(qp);
        psns_.push_back(random() & 0xffffff);
        ibv_qp_attr attr{};
        attr.qp_state = IBV_QPS_INIT;
        attr.pkey_index = 0;
        attr.port_num = port_;
        attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;
        require(ibv_modify_qp(qp, &attr,
                              IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT |
                                  IBV_QP_ACCESS_FLAGS) == 0,
                "Cannot initialize RC queue pair");
        for (int r = 0; r < kReceives; ++r) post_receive(q * kReceives + r);
      }
    } catch (...) {
      release();
      throw;
    }
  }

  ~Endpoint() { close(); }

  py::dict info() const {
    std::vector<uint32_t> numbers;
    for (auto* qp : qps_) numbers.push_back(qp->qp_num);
    return py::dict(
        "protocol"_a = 1,
        "gid"_a = py::bytes(reinterpret_cast<const char*>(&gid_), 16),
        "qps"_a = numbers, "psns"_a = psns_,
        "mtu"_a = static_cast<int>(port_attr_.active_mtu));
  }

  void connect(const py::dict& remote) {
    require(!connected_ && !closed_, "Endpoint already connected or closed");
    require(remote["protocol"].cast<int>() == 1,
            "Peer uses an incompatible MegaMoE RDMA protocol");
    std::string gid = remote["gid"].cast<std::string>();
    auto numbers = remote["qps"].cast<std::vector<uint32_t>>();
    auto psns = remote["psns"].cast<std::vector<uint32_t>>();
    require(gid.size() == 16 && numbers.size() == qps_.size() &&
                psns.size() == qps_.size(),
            "Peer queue-pair configuration differs");
    for (int q = 0; q < count_; ++q) {
      ibv_qp_attr attr{};
      attr.qp_state = IBV_QPS_RTR;
      attr.path_mtu = static_cast<ibv_mtu>(std::min(
          static_cast<int>(port_attr_.active_mtu), remote["mtu"].cast<int>()));
      attr.dest_qp_num = numbers[q];
      attr.rq_psn = psns[q];
      attr.max_dest_rd_atomic = 1;
      attr.min_rnr_timer = 12;
      attr.ah_attr.is_global = 1;
      attr.ah_attr.port_num = port_;
      attr.ah_attr.grh.sgid_index = gid_index_;
      attr.ah_attr.grh.hop_limit = 64;
      attr.ah_attr.grh.traffic_class = traffic_class_;
      std::memcpy(&attr.ah_attr.grh.dgid, gid.data(), 16);
      require(ibv_modify_qp(qps_[q], &attr,
                            IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
                                IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                IBV_QP_MAX_DEST_RD_ATOMIC |
                                IBV_QP_MIN_RNR_TIMER) == 0,
              "Cannot move RC queue pair to RTR");
      attr = {};
      attr.qp_state = IBV_QPS_RTS;
      attr.timeout = 14;
      attr.retry_cnt = 7;
      attr.rnr_retry = 7;
      attr.sq_psn = psns_[q];
      attr.max_rd_atomic = 1;
      require(ibv_modify_qp(qps_[q], &attr,
                            IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                                IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                IBV_QP_MAX_QP_RD_ATOMIC) == 0,
              "Cannot move RC queue pair to RTS");
    }
    connected_ = true;
    worker_ = std::thread([this] { progress(); });
  }

  py::dict register_job(uint64_t send, uint64_t receive,
                        size_t allocation_bytes, size_t peer_bytes) {
    require(connected_ && !closed_,
            "Connect endpoint before registering buffers");
    require(peer_bytes >= static_cast<size_t>(count_) &&
                peer_bytes <= std::numeric_limits<uint32_t>::max() &&
                allocation_bytes >= 2 * peer_bytes,
            "Invalid peer-major buffer capacity");
    auto job = std::make_unique<Job>();
    job->send = send;
    job->receive = receive;
    job->bytes = peer_bytes;
    job->send_region = std::make_unique<Region>(pd_, send, allocation_bytes);
    job->receive_region =
        std::make_unique<Region>(pd_, receive, allocation_bytes);
    cuda_check(
        cuMemHostAlloc(reinterpret_cast<void**>(&job->flags), sizeof(Flags),
                       CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
        "allocate stream handshake flags");
    std::memset(job->flags, 0, sizeof(Flags));
    cuda_check(cuMemHostGetDevicePointer(&job->device_flags, job->flags, 0),
               "map stream handshake flags");
    std::lock_guard<std::mutex> lock(mutex_);
    require(connected_ && !closed_,
            "Connect endpoint before registering buffers");
    require(jobs_.size() < (1u << 24), "Too many registered transfers");
    auto id = jobs_.size();
    auto result = py::dict("id"_a = id, "address"_a = receive,
                           "key"_a = job->receive_region->mr->rkey,
                           "bytes"_a = peer_bytes);
    jobs_.push_back(std::move(job));
    return result;
  }

  void connect_job(uint32_t id, const py::dict& remote) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& job = *jobs_.at(id);
    require(!job.connected && remote["id"].cast<uint32_t>() == id &&
                remote["bytes"].cast<size_t>() == job.bytes,
            "Peer transfer order or sizes differ");
    job.remote_receive = remote["address"].cast<uint64_t>();
    job.remote_key = remote["key"].cast<uint32_t>();
    job.connected = true;
  }

  void submit(uint32_t id, uint64_t stream_pointer) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& job = *jobs_.at(id);
    require(job.connected && !closed_, "Transfer is not connected");
    auto stream = reinterpret_cast<CUstream>(stream_pointer);
    cuda_check(
        cuMemcpyDtoDAsync(job.receive + self_ * job.bytes,
                          job.send + self_ * job.bytes, job.bytes, stream),
        "copy self contribution");
    // The default write includes a stream memory fence: the NIC cannot read
    // send storage until earlier packing/GEMM work is globally visible.
    cuda_check(cuStreamWriteValue32(stream,
                                    job.device_flags + offsetof(Flags, request),
                                    1, CU_STREAM_WRITE_VALUE_DEFAULT),
               "publish RDMA request");
  }

  void wait(uint32_t id, uint64_t stream_pointer) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& job = *jobs_.at(id);
    require(job.connected && !closed_, "Transfer is not connected");
    auto stream = reinterpret_cast<CUstream>(stream_pointer);
    auto done = job.device_flags + offsetof(Flags, done);
    cuda_check(cuStreamWaitValue32(stream, done, 1, CU_STREAM_WAIT_VALUE_EQ),
               "wait for RDMA completion");
    cuda_check(
        cuStreamWriteValue32(stream, done, 0, CU_STREAM_WRITE_VALUE_DEFAULT),
        "acknowledge RDMA completion");
  }

  void set_tracing(bool enabled) {
    std::lock_guard<std::mutex> lock(mutex_);
    tracing_ = enabled;
    traces_.clear();
  }
  py::list trace_snapshot() {
    std::lock_guard<std::mutex> lock(mutex_);
    py::list result;
    for (const auto& t : traces_)
      result.append(
          py::dict("job"_a = t.job, "epoch"_a = t.epoch,
                   "request_ns"_a = t.request_ns, "post_ns"_a = t.post_ns,
                   "complete_ns"_a = t.complete_ns, "bytes"_a = t.bytes));
    return result;
  }

  // Caller must finish GPU work and destroy captured graphs first. In
  // particular, deregistration cannot race an outstanding NIC DMA.
  void close() {
    if (closed_) return;
    stopping_.store(true);
    if (worker_.joinable()) worker_.join();
    release();
    closed_ = true;
  }

 private:
  void post_receive(size_t index) {
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(&credits_[index]);
    sge.length = sizeof(Credit);
    sge.lkey = credit_mr_->lkey;
    ibv_recv_wr wr{}, *bad = nullptr;
    wr.wr_id = index;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    require(ibv_post_recv(qps_[index / kReceives], &wr, &bad) == 0,
            "Cannot post receive credit");
  }
  void advertise(uint32_t id, Job& job) {
    Credit credit{job.epoch, id, 0};
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(&credit);
    sge.length = sizeof(credit);
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = kControlSend | id;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    require(ibv_post_send(qps_[0], &wr, &bad) == 0,
            "Cannot advertise receive credit");
  }
  void post_data(uint32_t id, Job& job) {
    job.post_ns = now_ns();
    job.posted = true;
    for (int q = 0; q < count_; ++q) {
      size_t begin = job.bytes * q / count_;
      size_t end = job.bytes * (q + 1) / count_;
      ibv_sge sge{};
      sge.addr = job.send + (1 - self_) * job.bytes + begin;
      sge.length = end - begin;
      sge.lkey = job.send_region->mr->lkey;
      ibv_send_wr wr{}, *bad = nullptr;
      wr.wr_id = (uint64_t{id} << 8) | q;
      wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
      wr.send_flags = IBV_SEND_SIGNALED;
      wr.imm_data = htonl((id << 8) | q);
      wr.sg_list = &sge;
      wr.num_sge = 1;
      wr.wr.rdma.remote_addr = job.remote_receive + self_ * job.bytes + begin;
      wr.wr.rdma.rkey = job.remote_key;
      require(ibv_post_send(qps_[q], &wr, &bad) == 0,
              "Cannot post GPU RDMA write");
    }
  }
  void poll() {
    ibv_wc completions[32];
    int count = ibv_poll_cq(cq_, 32, completions);
    require(count >= 0, "Cannot poll RDMA completions");
    for (int i = 0; i < count; ++i) {
      const auto& wc = completions[i];
      require(wc.status == IBV_WC_SUCCESS,
              "RDMA completion failure: " +
                  std::string(ibv_wc_status_str(wc.status)) +
                  ", vendor=" + std::to_string(wc.vendor_err));
      if (wc.opcode == IBV_WC_RECV) {
        require(wc.byte_len == sizeof(Credit), "Invalid receive-credit length");
        auto credit = credits_.at(wc.wr_id);
        auto& job = *jobs_.at(credit.job);
        require(credit.epoch == job.peer_epoch + 1,
                "Duplicate or out-of-order receive credit");
        job.peer_epoch = credit.epoch;
        post_receive(wc.wr_id);
      } else if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        auto tag = ntohl(wc.imm_data);
        auto& job = *jobs_.at(tag >> 8);
        unsigned int q = tag & 255;
        require(job.active && q < static_cast<unsigned>(count_) &&
                    !(job.received_mask & (1u << q)),
                "Unexpected or duplicate RDMA receive");
        job.received_mask |= 1u << q;
        post_receive(wc.wr_id);
      } else if (!(wc.wr_id & kControlSend)) {
        auto& job = *jobs_.at(wc.wr_id >> 8);
        unsigned int q = wc.wr_id & 255;
        require(job.active && q < static_cast<unsigned>(count_) &&
                    !(job.sent_mask & (1u << q)),
                "Unexpected or duplicate RDMA send completion");
        job.sent_mask |= 1u << q;
      }
    }
  }
  void progress() noexcept {
    try {
      cuda_check(cuCtxSetCurrent(cuda_context_), "set proxy CUDA context");
      while (!stopping_.load(std::memory_order_relaxed)) {
        {
          std::lock_guard<std::mutex> lock(mutex_);
          poll();
          for (uint32_t id = 0; id < jobs_.size(); ++id) {
            auto& job = *jobs_[id];
            if (!job.connected) continue;
            if (!job.active && load(&job.flags->request)) {
              require(!load(&job.flags->done),
                      "Transfer reused before GPU acknowledgement");
              require(job.epoch != std::numeric_limits<uint64_t>::max(),
                      "RDMA epoch overflow");
              ++job.epoch;
              job.active = true;
              job.request_ns = now_ns();
              advertise(id, job);
            }
            if (!job.active) continue;
            require(now_ns() - job.request_ns <= timeout_ns_,
                    "RDMA timeout at job " + std::to_string(id) + ", epoch " +
                        std::to_string(job.epoch) + ", peer epoch " +
                        std::to_string(job.peer_epoch) + ", send mask " +
                        std::to_string(job.sent_mask) + ", receive mask " +
                        std::to_string(job.received_mask));
            // Both endpoints grant a credit only after their GPU consumed the
            // previous receive and finished producing the next send.
            if (!job.posted && job.peer_epoch == job.epoch) post_data(id, job);
            unsigned int all = (1u << count_) - 1;
            if (job.sent_mask != all || job.received_mask != all) continue;
            if (flush_)
              cuda_check(cuFlushGPUDirectRDMAWrites(
                             CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                             CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER),
                         "flush GPUDirect receives");
            if (tracing_ && traces_.size() < 100000)
              traces_.push_back({job.epoch, job.request_ns, job.post_ns,
                                 now_ns(), job.bytes, id});
            job.active = job.posted = false;
            job.sent_mask = job.received_mask = 0;
            store(&job.flags->request, 0);
            store(&job.flags->done, 1);
          }
        }
        std::this_thread::yield();
      }
    } catch (const std::exception& error) {
      // A captured GPU wait cannot propagate a C++ exception to Python. Never
      // unblock it with invalid data: fail this worker and let its supervisor
      // terminate the distributed step. Peers have the same bounded watchdog.
      std::fprintf(stderr, "MegaMoE RDMA proxy failed: %s\n", error.what());
      std::fflush(stderr);
      std::_Exit(EXIT_FAILURE);
    }
  }
  void release() {
    for (auto* qp : qps_) ibv_destroy_qp(qp);
    qps_.clear();
    jobs_.clear();
    if (credit_mr_) ibv_dereg_mr(credit_mr_);
    if (cq_) ibv_destroy_cq(cq_);
    if (pd_) ibv_dealloc_pd(pd_);
    if (context_) ibv_close_device(context_);
    credit_mr_ = nullptr;
    cq_ = nullptr;
    pd_ = nullptr;
    context_ = nullptr;
  }

  int port_, gid_index_, count_, traffic_class_, self_;
  uint64_t timeout_ns_;
  CUcontext cuda_context_ = nullptr;
  ibv_context* context_ = nullptr;
  ibv_pd* pd_ = nullptr;
  ibv_cq* cq_ = nullptr;
  ibv_mr* credit_mr_ = nullptr;
  ibv_port_attr port_attr_{};
  ibv_gid gid_{};
  std::vector<ibv_qp*> qps_;
  std::vector<uint32_t> psns_;
  std::vector<Credit> credits_;
  std::vector<std::unique_ptr<Job>> jobs_;
  std::vector<Trace> traces_;
  std::mutex mutex_;
  std::thread worker_;
  std::atomic<bool> stopping_{false};
  bool connected_ = false, closed_ = false, flush_ = false, tracing_ = false;
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<Endpoint>(module, "Endpoint")
      .def(py::init<const std::string&, int, int, int, int, int, int>())
      .def("info", &Endpoint::info)
      .def("connect", &Endpoint::connect)
      .def("register_job", &Endpoint::register_job)
      .def("connect_job", &Endpoint::connect_job)
      .def("submit", &Endpoint::submit)
      .def("wait", &Endpoint::wait)
      .def("set_tracing", &Endpoint::set_tracing)
      .def("trace_snapshot", &Endpoint::trace_snapshot)
      .def("close", &Endpoint::close, py::call_guard<py::gil_scoped_release>());
}
