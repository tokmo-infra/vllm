ARG BASE_IMAGE=sha256:2c91fd527d05992e2b373938c44556fc94259ce6a6069a7158c2d1f2568cd828
FROM ${BASE_IMAGE}
COPY integrations/glm53flash/release.json /opt/glm53flash-1103-release.json
RUN python3 -c "import hashlib,json,pathlib; m=json.loads(pathlib.Path('/opt/glm53flash-1103-release.json').read_text()); r=pathlib.Path('/usr/local/lib/python3.12/dist-packages'); bad=[x['path'] for x in m['base_validation_files'] if hashlib.sha256((r/x['path']).read_bytes()).hexdigest()!=x['sha256']]; assert not bad, bad"
COPY vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
COPY vllm/envs.py /usr/local/lib/python3.12/dist-packages/vllm/envs.py
COPY vllm/v1/core/kv_cache_manager.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py
COPY vllm/v1/core/sched/scheduler.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
COPY vllm/v1/core/sched/hot_tail_budget.py /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/hot_tail_budget.py
ENV VLLM_EXPERIMENTAL_HOT_TAIL_ADMISSION=0 \
    VLLM_HOT_TAIL_ADMISSION_DIAGNOSTICS=0
RUN python3 -m compileall -q \
    /usr/local/lib/python3.12/dist-packages/vllm/envs.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/hot_tail_budget.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py
LABEL glm.release="glm5.3-flash-1103"
