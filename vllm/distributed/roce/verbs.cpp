// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Multi-host GPUDirect transport. CUDA stream memory operations drive a CPU
// verbs proxy; no communication kernel competes with native MegaMoE's grid.
#include <cuda.h>
#include <ATen/dlpack.h>
#include <infiniband/verbs.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arpa/inet.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
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

#include "protocol.h"

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

// CUDA IPC provides peer mappings for the cuMemAlloc storage required by this
// DMA-BUF path. PyTorch's VMM symmetric allocator does not request RDMA-capable
// allocations; do not silently ignore its unsupported SYNC_MEMOPS attribute.
class PeerAllocation : public std::enable_shared_from_this<PeerAllocation> {
 public:
  explicit PeerAllocation(int64_t bytes) : bytes_(bytes) {
    require(bytes > 0 && bytes % kPage == 0 && bytes < (int64_t{1} << 32),
            "Peer storage must have a positive 64-KiB-padded size below 4 GiB");
    cuda_check(cuCtxGetCurrent(&context_), "get peer-storage context");
    require(context_ != nullptr, "Initialize CUDA before peer allocation");
    cuda_check(cuCtxGetDevice(&device_), "get peer-storage device");
    cuda_check(cuMemAlloc(&base_, bytes + kPage), "allocate peer RDMA storage");
    address_ = (base_ + kPage - 1) / kPage * kPage;
  }
  ~PeerAllocation() {
    if (cuCtxPushCurrent(context_) != CUDA_SUCCESS) return;
    cuCtxSynchronize();
    for (auto ptr : imported_) cuIpcCloseMemHandle(ptr);
    if (base_) cuMemFree(base_);
    CUcontext previous;
    cuCtxPopCurrent(&previous);
  }
  py::dict info() const {
    CUipcMemHandle handle;
    cuda_check(cuIpcGetMemHandle(&handle, base_), "export peer RDMA storage");
    return py::dict("handle"_a = py::bytes(reinterpret_cast<char*>(&handle), sizeof(handle)),
                    "offset"_a = address_ - base_, "bytes"_a = bytes_);
  }
  void connect(const py::list& peers, uint32_t rank) {
    require(addresses_.empty() && rank < peers.size(), "Invalid peer-storage rendezvous");
    for (uint32_t peer = 0; peer < peers.size(); ++peer) {
      auto metadata = peers[peer].cast<py::dict>();
      auto raw = metadata["handle"].cast<std::string>();
      auto offset = metadata["offset"].cast<uint64_t>();
      require(raw.size() == sizeof(CUipcMemHandle) && offset < kPage &&
                  metadata["bytes"].cast<int64_t>() == bytes_,
              "Peer-storage layouts differ");
      if (peer == rank) {
        addresses_.push_back(address_);
      } else {
        CUipcMemHandle handle;
        std::memcpy(&handle, raw.data(), sizeof(handle));
        CUdeviceptr mapped;
        cuda_check(cuIpcOpenMemHandle(&mapped, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
                   "import peer RDMA storage");
        imported_.push_back(mapped);
        addresses_.push_back(mapped + offset);
      }
    }
  }
  std::vector<uint64_t> addresses() const { return addresses_; }
  py::capsule view(int peer) {
    struct View {
      DLManagedTensor tensor{};
      std::shared_ptr<PeerAllocation> owner;
      int64_t shape;
    };
    auto address = peer < 0 ? address_ : addresses_.at(peer);
    auto* view = new View;
    view->owner = shared_from_this();
    view->shape = bytes_;
    view->tensor.manager_ctx = view;
    view->tensor.dl_tensor = {reinterpret_cast<void*>(address), {kDLCUDA, device_},
                              1, {kDLUInt, 8, 1}, &view->shape, nullptr, 0};
    view->tensor.deleter = [](DLManagedTensor* tensor) {
      delete static_cast<View*>(tensor->manager_ctx);
    };
    return py::capsule(&view->tensor, "dltensor", [](PyObject* capsule) {
      if (PyCapsule_IsValid(capsule, "dltensor")) {
        auto* tensor = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
        tensor->deleter(tensor);
      }
    });
  }

 private:
  CUcontext context_ = nullptr;
  CUdevice device_ = 0;
  CUdeviceptr base_ = 0, address_ = 0;
  int64_t bytes_;
  std::vector<uint64_t> addresses_;
  std::vector<CUdeviceptr> imported_;
};

struct Region {
  ibv_mr* mr = nullptr;
  int fd = -1;
  bool relaxed_ordering = true;
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
    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;
    // Completion still guarantees visibility. Receive credits prevent
    // overlapping generations from writing the same region concurrently.
    mr = ibv_reg_dmabuf_mr(pd, 0, bytes, address, fd,
                           access | IBV_ACCESS_RELAXED_ORDERING);
    if (!mr && (errno == EINVAL || errno == EOPNOTSUPP)) {
      relaxed_ordering = false;
      mr = ibv_reg_dmabuf_mr(pd, 0, bytes, address, fd, access);
    }
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

struct HostSend {
  void* address = nullptr;
  ibv_mr* mr = nullptr;
  HostSend(ibv_pd* pd, size_t bytes) {
    cuda_check(cuMemHostAlloc(&address, bytes, CU_MEMHOSTALLOC_PORTABLE),
               "allocate pinned send staging");
    mr = ibv_reg_mr(pd, address, bytes, IBV_ACCESS_LOCAL_WRITE);
    if (!mr) {
      const auto error = std::string(std::strerror(errno));
      cuMemFreeHost(address);
      address = nullptr;
      throw std::runtime_error("register pinned send staging: " + error);
    }
  }
  ~HostSend() {
    if (mr) ibv_dereg_mr(mr);
    if (address) cuMemFreeHost(address);
  }
};

struct alignas(64) Flags {
  alignas(64) uint32_t request;
  alignas(64) uint32_t done;
  alignas(64) uint32_t receive_request;
  alignas(64) uint32_t receive_done;
  alignas(64) uint32_t send_rows;
  alignas(64) uint32_t receive_rows;
};
enum class MessageKind : uint32_t { Credit = 1, Payload = 2 };
struct Credit {
  uint64_t epoch;
  uint32_t job;
  uint32_t bytes;
  MessageKind kind;
  uint32_t reserved = 0;
};
struct Trace {
  uint64_t epoch, request_ns, post_ns, complete_ns, bytes;
  uint32_t job;
  bool receive;
  uint64_t cq_ns = 0, visibility_begin_ns = 0, visibility_end_ns = 0;
};
struct Job {
  Job(uint32_t capacity, uint32_t row, uint32_t qps)
      : bytes(capacity), row_bytes(row), state(capacity, row, qps) {}
  std::unique_ptr<Region> send_region, receive_region;
  std::unique_ptr<HostSend> host_send;
  CUevent produced = nullptr, staged = nullptr;
  uint64_t send, receive, remote_receive = 0;
  uint32_t remote_key = 0;
  size_t bytes, row_bytes;
  vllm::roce::ExchangeState state;
  Flags* flags = nullptr;
  CUdeviceptr device_flags = 0;
  bool connected = false, copy_self = true, visibility_pending = false;
  std::atomic<bool> visibility_ready{false};
  uint64_t receive_cq_ns = 0, visibility_begin_ns = 0, visibility_end_ns = 0;
  uint64_t request_ns = 0, receive_request_ns = 0, post_ns = 0;
  ~Job() {
    if (produced) cuEventDestroy(produced);
    if (staged) cuEventDestroy(staged);
    if (flags) cuMemFreeHost(flags);
  }
};

class Endpoint {
 public:
  Endpoint(const std::string& hca, int port, int gid_index, int qps,
           int traffic_class, int timeout_seconds, int self, bool host_staging,
           int peer = -1, int world = 2, bool copy_self = true,
           bool peer_reads = false)
      : port_(port),
        gid_index_(gid_index),
        count_(qps),
        traffic_class_(traffic_class),
        self_(self),
        peer_(peer < 0 ? 1 - self : peer),
        world_(world),
        copy_self_(copy_self),
        timeout_ns_(uint64_t(timeout_seconds) * 1000000000) {
    require(qps >= 1 && qps <= 8 && world >= 2 && self >= 0 && self < world &&
                peer_ >= 0 && peer_ < world && peer_ != self,
            "Expected 1..8 QPs and distinct valid host lane ranks");
    require(port >= 1 && port <= 255 && gid_index >= 0 && gid_index <= 255 &&
                traffic_class >= 0 && traffic_class <= 255 &&
                timeout_seconds > 0,
            "Invalid RDMA configuration");
    try {
      cuda_check(cuCtxGetCurrent(&cuda_context_), "get current CUDA context");
      require(cuda_context_ != nullptr, "Initialize CUDA before RDMA");
      if (host_staging)
        cuda_check(cuStreamCreate(&staging_stream_, CU_STREAM_NON_BLOCKING),
                   "create send staging stream");
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
      flush_scope_ = peer_reads
                         ? CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES
                         : CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER;
      flush_ = ordering < (peer_reads ? CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES
                                    : CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER);
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
          ibv_create_cq(context_, (count_ + 1) * kReceives * 2, nullptr, nullptr, 0);
      require(cq_ != nullptr, "Cannot create completion queue");
      credits_.resize((count_ + 1) * kReceives);
      credit_mr_ =
          ibv_reg_mr(pd_, credits_.data(), credits_.size() * sizeof(Credit),
                     IBV_ACCESS_LOCAL_WRITE);
      require(credit_mr_ != nullptr, "Cannot register control receive memory");
      std::random_device random;
      // The last QP carries only credits/descriptors, so payload WRs cannot
      // delay the control traffic needed by the opposite direction.
      for (int q = 0; q <= count_; ++q) {
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
        "protocol"_a = 4,
        "gid"_a = py::bytes(reinterpret_cast<const char*>(&gid_), 16),
        "qps"_a = numbers, "psns"_a = psns_,
        "mtu"_a = static_cast<int>(port_attr_.active_mtu));
  }

  void connect(const py::dict& remote) {
    require(!connected_ && !closed_, "Endpoint already connected or closed");
    require(remote["protocol"].cast<int>() == 4,
            "Peer uses an incompatible MegaMoE RDMA protocol");
    std::string gid = remote["gid"].cast<std::string>();
    auto numbers = remote["qps"].cast<std::vector<uint32_t>>();
    auto psns = remote["psns"].cast<std::vector<uint32_t>>();
    require(gid.size() == 16 && numbers.size() == qps_.size() &&
                psns.size() == qps_.size(),
            "Peer queue-pair configuration differs");
    for (int q = 0; q <= count_; ++q) {
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
    if (flush_) visibility_worker_ = std::thread([this] { make_visible(); });
    worker_ = std::thread([this] { progress(); });
  }

  py::dict register_job(uint64_t send, uint64_t receive,
                        size_t allocation_bytes, size_t peer_bytes,
                        size_t row_bytes, bool copy_self = true,
                        bool acquire_before_send = false) {
    require(connected_ && !closed_,
            "Connect endpoint before registering buffers");
    require(peer_bytes >= static_cast<size_t>(count_) &&
                peer_bytes <= std::numeric_limits<uint32_t>::max() &&
                allocation_bytes >= world_ * peer_bytes && row_bytes > 0 &&
                peer_bytes % row_bytes == 0,
            "Invalid peer-major buffer capacity");
    auto job = std::make_unique<Job>(peer_bytes, row_bytes, count_);
    job->send = send;
    job->receive = receive;
    job->bytes = peer_bytes;
    job->row_bytes = row_bytes;
    job->copy_self = copy_self;
    if (staging_stream_) {
      job->host_send = std::make_unique<HostSend>(pd_, peer_bytes);
      cuda_check(cuEventCreate(&job->produced, CU_EVENT_DISABLE_TIMING),
                 "create send producer event");
      cuda_check(cuEventCreate(&job->staged, CU_EVENT_DISABLE_TIMING),
                 "create send staged event");
    } else {
      job->send_region = std::make_unique<Region>(pd_, send, allocation_bytes);
    }
    job->receive_region =
        std::make_unique<Region>(pd_, receive, allocation_bytes);
    cuda_check(
        cuMemHostAlloc(reinterpret_cast<void**>(&job->flags), sizeof(Flags),
                       CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP),
        "allocate stream handshake flags");
    std::memset(job->flags, 0, sizeof(Flags));
    // Pooled producers acquire even the first generation. Later acquisitions
    // consume the preceding NIC completion before overwriting this slot.
    job->flags->done = acquire_before_send ? 1 : 0;
    cuda_check(cuMemHostGetDevicePointer(&job->device_flags, job->flags, 0),
               "map stream handshake flags");
    auto lock = lock_api();
    require(connected_ && !closed_,
            "Connect endpoint before registering buffers");
    require(jobs_.size() < (1u << 24), "Too many registered transfers");
    auto id = jobs_.size();
    auto result =
        py::dict("id"_a = id, "address"_a = receive,
                 "key"_a = job->receive_region->mr->rkey,
                 "bytes"_a = peer_bytes, "row_bytes"_a = row_bytes,
                 "request_address"_a = job->device_flags + offsetof(Flags, request),
                 "send_done_address"_a = job->device_flags + offsetof(Flags, done),
                 "receive_request_address"_a = job->device_flags + offsetof(Flags, receive_request),
                 "receive_done_address"_a = job->device_flags + offsetof(Flags, receive_done),
                 "send_rows_address"_a = job->device_flags + offsetof(Flags, send_rows),
                 "receive_rows_address"_a = job->device_flags + offsetof(Flags, receive_rows),
                 "relaxed_ordering"_a = job->receive_region->relaxed_ordering,
                 "host_staging"_a = staging_stream_ != nullptr);
    jobs_.push_back(std::move(job));
    return result;
  }

  void connect_job(uint32_t id, const py::dict& remote) {
    auto lock = lock_api();
    auto& job = *jobs_.at(id);
    require(!job.connected && remote["id"].cast<uint32_t>() == id &&
                remote["bytes"].cast<size_t>() == job.bytes &&
                remote["row_bytes"].cast<size_t>() == job.row_bytes,
            "Peer transfer order or sizes differ");
    job.remote_receive = remote["address"].cast<uint64_t>();
    job.remote_key = remote["key"].cast<uint32_t>();
    job.connected = true;
  }

  void submit(uint32_t id, uint64_t stream_pointer, uint64_t rows_pointer) {
    reserve(id, stream_pointer);
    publish(id, stream_pointer, rows_pointer);
  }

  void reserve(uint32_t id, uint64_t stream_pointer) {
    CUdeviceptr request;
    {
      auto lock = lock_api();
      const auto& job = *jobs_.at(id);
      require(job.connected && !closed_, "Transfer is not connected");
      request = job.device_flags + offsetof(Flags, receive_request);
    }
    cuda_check(cuStreamWriteValue32(reinterpret_cast<CUstream>(stream_pointer),
                                    request, 1, CU_STREAM_WRITE_VALUE_DEFAULT),
               "grant receive storage");
  }

  void publish(uint32_t id, uint64_t stream_pointer, uint64_t rows_pointer) {
    uint64_t send, receive, request, rows_device;
    uint32_t* rows_host;
    size_t bytes, row_bytes;
    bool copy_self;
    void* host_send = nullptr;
    CUevent produced = nullptr, staged = nullptr;
    {
      auto lock = lock_api();
      const auto& job = *jobs_.at(id);
      require(job.connected && !closed_, "Transfer is not connected");
      send = job.send;
      receive = job.receive;
      bytes = job.bytes;
      row_bytes = job.row_bytes;
      copy_self = copy_self_ && job.copy_self;
      request = job.device_flags + offsetof(Flags, request);
      rows_device = job.device_flags + offsetof(Flags, send_rows);
      rows_host = &job.flags->send_rows;
      if (job.host_send) {
        host_send = job.host_send->address;
        produced = job.produced;
        staged = job.staged;
      }
    }
    // CUDA enqueue calls can block when the stream command queue is full.
    // The proxy must still drain RDMA while the GPU waits for a completion.
    auto stream = reinterpret_cast<CUstream>(stream_pointer);
    if (copy_self)
      cuda_check(cuMemcpyDtoDAsync(receive + self_ * bytes,
                                   send + self_ * bytes, bytes, stream),
                 "copy self contribution");
    auto publish_stream = stream;
    if (rows_pointer)
      cuda_check(
          cuMemcpyDtoHAsync(rows_host, rows_pointer, sizeof(uint32_t), stream),
          "copy outgoing row count");
    else
      cuda_check(cuStreamWriteValue32(stream, rows_device, bytes / row_bytes,
                                      CU_STREAM_WRITE_VALUE_DEFAULT),
                 "publish full outgoing row count");
    if (host_send) {
      cuda_check(cuEventRecord(produced, stream), "record send producer");
      cuda_check(cuStreamWaitEvent(staging_stream_, produced, 0),
                 "wait for send producer");
      cuda_check(cuMemcpyDtoHAsync(host_send, send + peer_ * bytes, bytes,
                                   staging_stream_),
                 "stage outgoing payload");
      publish_stream = staging_stream_;
    }
    // The default write includes a stream memory fence: the NIC cannot read
    // send storage until earlier packing/GEMM work is globally visible.
    cuda_check(cuStreamWriteValue32(publish_stream, request, 1,
                                    CU_STREAM_WRITE_VALUE_DEFAULT),
               "publish RDMA request");
    if (staged)
      cuda_check(cuEventRecord(staged, staging_stream_), "record staged send");
  }

  void wait(uint32_t id, uint64_t stream_pointer, uint64_t rows_pointer) {
    wait_receive(id, stream_pointer, rows_pointer);
    wait_send(id, stream_pointer);
  }

  void wait_send(uint32_t id, uint64_t stream_pointer) {
    CUdeviceptr done;
    CUevent staged;
    {
      auto lock = lock_api();
      const auto& job = *jobs_.at(id);
      require(job.connected && !closed_, "Transfer is not connected");
      done = job.device_flags + offsetof(Flags, done);
      staged = job.staged;
    }
    auto stream = reinterpret_cast<CUstream>(stream_pointer);
    if (staged)
      cuda_check(cuStreamWaitEvent(stream, staged, 0), "join send staging");
    cuda_check(cuStreamWaitValue32(stream, done, 1, CU_STREAM_WAIT_VALUE_EQ),
               "wait for RDMA send completion");
    cuda_check(cuStreamWriteValue32(stream, done, 0, CU_STREAM_WRITE_VALUE_DEFAULT),
               "release RDMA send completion");
  }

  void wait_receive(uint32_t id, uint64_t stream_pointer, uint64_t rows_pointer) {
    CUdeviceptr done;
    const uint32_t* rows_host;
    {
      auto lock = lock_api();
      const auto& job = *jobs_.at(id);
      require(job.connected && !closed_, "Transfer is not connected");
      done = job.device_flags + offsetof(Flags, receive_done);
      rows_host = &job.flags->receive_rows;
    }
    auto stream = reinterpret_cast<CUstream>(stream_pointer);
    cuda_check(cuStreamWaitValue32(stream, done, 1, CU_STREAM_WAIT_VALUE_EQ),
               "wait for RDMA receive completion");
    if (rows_pointer)
      cuda_check(
          cuMemcpyHtoDAsync(rows_pointer, rows_host, sizeof(uint32_t), stream),
          "copy incoming row count");
    cuda_check(
        cuStreamWriteValue32(stream, done, 0, CU_STREAM_WRITE_VALUE_DEFAULT),
        "acknowledge RDMA completion");
  }

  void set_tracing(bool enabled) {
    auto lock = lock_api();
    tracing_ = enabled;
    traces_.clear();
  }
  py::list trace_snapshot() {
    auto lock = lock_api();
    py::list result;
    for (const auto& t : traces_)
      result.append(
          py::dict("job"_a = t.job, "epoch"_a = t.epoch,
                   "request_ns"_a = t.request_ns, "post_ns"_a = t.post_ns,
                   "complete_ns"_a = t.complete_ns, "bytes"_a = t.bytes,
                   "direction"_a = t.receive ? "receive" : "send",
                   "cq_ns"_a = t.cq_ns,
                   "visibility_begin_ns"_a = t.visibility_begin_ns,
                   "visibility_end_ns"_a = t.visibility_end_ns));
    return result;
  }

  // Caller must finish GPU work and destroy captured graphs first. In
  // particular, deregistration cannot race an outstanding NIC DMA.
  void close() {
    if (closed_) return;
    stopping_.store(true);
    if (worker_.joinable()) worker_.join();
    {
      std::lock_guard<std::mutex> lock(visibility_mutex_);
      visibility_stopping_ = true;
    }
    visibility_changed_.notify_one();
    if (visibility_worker_.joinable()) visibility_worker_.join();
    release();
    closed_ = true;
  }

 private:
  std::unique_lock<std::mutex> lock_api() {
    // std::mutex is not fair. A busy proxy must not repeatedly reacquire it
    // while the model thread is trying to enqueue the next rank's GPU work.
    // Otherwise peer resident kernels can wait for a rank starved on this lock.
    api_waiters_.fetch_add(1, std::memory_order_acq_rel);
    try {
      std::unique_lock<std::mutex> lock(mutex_);
      api_waiters_.fetch_sub(1, std::memory_order_release);
      return lock;
    } catch (...) {
      api_waiters_.fetch_sub(1, std::memory_order_release);
      throw;
    }
  }

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
  void advertise(uint32_t id, uint64_t epoch, MessageKind kind,
                 uint32_t bytes = 0) {
    Credit credit{epoch, id, bytes, kind};
    ibv_sge sge{};
    sge.addr = reinterpret_cast<uint64_t>(&credit);
    sge.length = sizeof(credit);
    ibv_send_wr wr{}, *bad = nullptr;
    wr.wr_id = kControlSend | id;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_INLINE | IBV_SEND_SIGNALED;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    require(ibv_post_send(qps_.back(), &wr, &bad) == 0,
            "Cannot advertise receive credit");
  }
  void post_data(uint32_t id, Job& job) {
    job.post_ns = now_ns();
    job.state.mark_posted();
    for (int q = 0; q < count_; ++q) {
      size_t begin = uint64_t(job.state.send_bytes) * q / count_;
      size_t end = uint64_t(job.state.send_bytes) * (q + 1) / count_;
      if (begin == end) continue;
      ibv_sge sge{};
      sge.addr =
          job.host_send
              ? reinterpret_cast<uint64_t>(job.host_send->address) + begin
              : job.send + peer_ * job.bytes + begin;
      sge.length = end - begin;
      sge.lkey =
          job.host_send ? job.host_send->mr->lkey : job.send_region->mr->lkey;
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
        if (credit.kind == MessageKind::Credit) {
          require(credit.bytes == 0, "Receive credit carries a payload length");
          job.state.accept_credit(credit.epoch);
        } else {
          require(credit.kind == MessageKind::Payload, "Unknown control message");
          job.state.accept_descriptor(credit.epoch, credit.bytes);
        }
        post_receive(wc.wr_id);
      } else if (wc.opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        auto tag = ntohl(wc.imm_data);
        auto& job = *jobs_.at(tag >> 8);
        unsigned int q = tag & 255;
        job.state.complete_receive_stripe(q);
        post_receive(wc.wr_id);
      } else if (!(wc.wr_id & kControlSend)) {
        auto& job = *jobs_.at(wc.wr_id >> 8);
        unsigned int q = wc.wr_id & 255;
        job.state.complete_send_stripe(q);
      }
    }
  }
  // A peer-visible flush may block in the CUDA driver. It must not hold the
  // verbs lock or prevent posting unrelated sends and receive credits.
  void make_visible() noexcept {
    try {
      cuda_check(cuCtxSetCurrent(cuda_context_), "set visibility CUDA context");
      std::vector<Job*> batch;
      while (true) {
        {
          std::unique_lock<std::mutex> lock(visibility_mutex_);
          visibility_changed_.wait(lock, [this] {
            return visibility_stopping_ || !visibility_queue_.empty();
          });
          if (visibility_queue_.empty() && visibility_stopping_) return;
          batch.swap(visibility_queue_);
        }
        const auto begin = now_ns();
        cuda_check(cuFlushGPUDirectRDMAWrites(
                       CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                       flush_scope_),
                   "flush GPUDirect receives");
        const auto end = now_ns();
        for (auto* job : batch) {
          job->visibility_begin_ns = begin;
          job->visibility_end_ns = end;
          job->visibility_ready.store(true, std::memory_order_release);
        }
        batch.clear();
      }
    } catch (const std::exception& error) {
      std::fprintf(stderr, "MegaMoE RDMA visibility failed: %s\n", error.what());
      std::fflush(stderr);
      std::_Exit(EXIT_FAILURE);
    }
  }
  void progress() noexcept {
    try {
      cuda_check(cuCtxSetCurrent(cuda_context_), "set proxy CUDA context");
      while (!stopping_.load(std::memory_order_relaxed)) {
        if (api_waiters_.load(std::memory_order_acquire)) {
          std::this_thread::yield();
          continue;
        }
        {
          std::lock_guard<std::mutex> lock(mutex_);
          poll();
          const uint32_t size = jobs_.size();
          uint32_t next_send = size;
          for (uint32_t step = 0; step < size; ++step) {
            const uint32_t id = (scan_start_ + step) % size;
            auto& job = *jobs_[id];
            if (!job.connected) continue;
            auto& state = job.state;
            if (!state.receiving && load(&job.flags->receive_request)) {
              require(!load(&job.flags->receive_done),
                      "Receive reused before GPU acknowledgement");
              state.begin_receive();
              job.receive_request_ns = now_ns();
              advertise(id, state.receive_epoch, MessageKind::Credit);
            }
            if (!state.sending && load(&job.flags->request)) {
              require(!load(&job.flags->done),
                      "Send reused before GPU acknowledgement");
              const uint64_t bytes =
                  uint64_t(load(&job.flags->send_rows)) * job.row_bytes;
              require(bytes <= job.bytes,
                      "Outgoing row count exceeds registered capacity");
              state.begin_send(bytes);
              job.request_ns = now_ns();
              advertise(id, state.send_epoch, MessageKind::Payload, bytes);
            }
            // Idle graph shapes need no clock read or error-string allocation.
            const uint64_t now = (state.sending || state.receiving) ? now_ns() : 0;
            if (state.sending && now - job.request_ns > timeout_ns_)
              throw std::runtime_error(
                  "RDMA send timeout at job " + std::to_string(id) +
                  ", epoch " + std::to_string(state.send_epoch));
            if (state.receiving && now - job.receive_request_ns > timeout_ns_)
              throw std::runtime_error(
                  "RDMA receive timeout at job " + std::to_string(id) +
                  ", epoch " + std::to_string(state.receive_epoch));
            if (state.can_post() && next_send == size) next_send = id;
            if (state.send_complete()) {
              if (tracing_ && traces_.size() < 100000)
                traces_.push_back({state.send_epoch, job.request_ns, job.post_ns,
                                   now_ns(), state.send_bytes, id, false});
              state.finish_send();
              store(&job.flags->request, 0);
              store(&job.flags->done, 1);
            }
            if (state.receive_complete() && !job.visibility_pending) {
              job.receive_cq_ns = now_ns();
              job.visibility_pending = true;
              job.visibility_begin_ns = job.visibility_end_ns = 0;
              job.visibility_ready.store(!flush_, std::memory_order_relaxed);
              if (flush_) {
                {
                  std::lock_guard<std::mutex> visibility_lock(visibility_mutex_);
                  visibility_queue_.push_back(&job);
                }
                visibility_changed_.notify_one();
              }
            }
            if (job.visibility_pending &&
                job.visibility_ready.load(std::memory_order_acquire)) {
              const uint32_t bytes = state.receive_bytes();
              if (tracing_ && traces_.size() < 100000)
                traces_.push_back({state.receive_epoch, job.receive_request_ns, 0,
                                   now_ns(), bytes, id, true, job.receive_cq_ns,
                                   job.visibility_begin_ns, job.visibility_end_ns});
              state.finish_receive();
              job.visibility_pending = false;
              store(&job.flags->receive_rows, bytes / job.row_bytes);
              store(&job.flags->receive_request, 0);
              store(&job.flags->receive_done, 1);
            }
          }
          // Visit every completion and credit before posting another payload.
          // Rotate the first candidate so ready returns and later input groups
          // receive service even while an earlier slot is repeatedly reused.
          if (next_send != size) {
            post_data(next_send, *jobs_[next_send]);
            scan_start_ = (next_send + 1) % size;
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
    if (staging_stream_) cuStreamDestroy(staging_stream_);
    staging_stream_ = nullptr;
    credit_mr_ = nullptr;
    cq_ = nullptr;
    pd_ = nullptr;
    context_ = nullptr;
  }

  int port_, gid_index_, count_, traffic_class_, self_, peer_, world_;
  bool copy_self_;
  uint64_t timeout_ns_;
  CUcontext cuda_context_ = nullptr;
  CUstream staging_stream_ = nullptr;
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
  uint32_t scan_start_ = 0;
  std::vector<Trace> traces_;
  std::mutex mutex_;
  std::atomic<uint32_t> api_waiters_{0};
  std::thread worker_, visibility_worker_;
  std::mutex visibility_mutex_;
  std::condition_variable visibility_changed_;
  std::vector<Job*> visibility_queue_;
  bool visibility_stopping_ = false;
  std::atomic<bool> stopping_{false};
  bool connected_ = false, closed_ = false, flush_ = false, tracing_ = false;
  CUflushGPUDirectRDMAWritesScope flush_scope_ =
      CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER;
};
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  py::class_<PeerAllocation, std::shared_ptr<PeerAllocation>>(module, "PeerAllocation")
      .def(py::init<int64_t>())
      .def("info", &PeerAllocation::info)
      .def("connect", &PeerAllocation::connect)
      .def("addresses", &PeerAllocation::addresses)
      .def("view", &PeerAllocation::view, py::arg("peer") = -1);
  py::class_<Endpoint>(module, "Endpoint")
      .def(py::init<const std::string&, int, int, int, int, int, int, bool, int,
                    int, bool, bool>(),
           py::arg("hca"), py::arg("port"), py::arg("gid_index"),
           py::arg("qps"), py::arg("traffic_class"), py::arg("timeout_seconds"),
           py::arg("self"), py::arg("host_staging"), py::arg("peer") = -1,
           py::arg("world") = 2, py::arg("copy_self") = true,
           py::arg("peer_reads") = false)
      .def("info", &Endpoint::info)
      .def("connect", &Endpoint::connect)
      .def("register_job", &Endpoint::register_job,
           py::arg("send"), py::arg("receive"), py::arg("allocation_bytes"),
           py::arg("peer_bytes"), py::arg("row_bytes"), py::arg("copy_self") = true,
           py::arg("acquire_before_send") = false)
      .def("connect_job", &Endpoint::connect_job)
      .def("submit", &Endpoint::submit)
      .def("reserve", &Endpoint::reserve)
      .def("publish", &Endpoint::publish)
      .def("wait", &Endpoint::wait)
      .def("wait_receive", &Endpoint::wait_receive)
      .def("wait_send", &Endpoint::wait_send)
      .def("set_tracing", &Endpoint::set_tracing)
      .def("trace_snapshot", &Endpoint::trace_snapshot)
      .def("close", &Endpoint::close, py::call_guard<py::gil_scoped_release>());
}
