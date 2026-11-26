# Environment Variables

vLLM uses the following environment variables to configure the system:

!!! warning
    Please note that `VLLM_PORT` and `VLLM_HOST_IP` set the port and ip for vLLM's **internal usage**. It is not the port and ip for the API server. If you use `--host $VLLM_HOST_IP` and `--port $VLLM_PORT` to start the API server, it will not work.

    Most vLLM-specific environment variables are prefixed with `VLLM_` (a handful of standard names — for example `CUDA_VISIBLE_DEVICES`, `MAX_JOBS`, `S3_ACCESS_KEY_ID`/`S3_SECRET_ACCESS_KEY`/`S3_ENDPOINT_URL`, `DO_NOT_TRACK`, `NO_COLOR` — are also read directly when set). **Special care should be taken for Kubernetes users**: please do not name the service as `vllm`, otherwise environment variables set by Kubernetes might conflict with vLLM's environment variables, because [Kubernetes sets environment variables for each service with the capitalized service name as the prefix](https://kubernetes.io/docs/concepts/services-networking/service/#environment-variables).

```python
--8<-- "vllm/envs.py:env-vars-definition"
```

## DeepSeek V4.1 frontend token cache

With the Python frontend and `--tokenizer-mode deepseek_v41`, set
`VLLM_DSV41_TOKENIZER_CACHE_GIB` to the retained CPU segment budget in GiB
per API process. The default is zero. Eight API processes with a budget of
12 GiB each may retain 96 GiB in addition to normal process memory.
Requests with the same cache salt can reuse text between tokenizer special
tokens. Requests needing offsets or multimodal handling use the native path.
The per-process `vllm:tokenizer_cache_tokens_total` metric distinguishes
encoded, reused and special tokens. For cold measurements, point
`VLLM_DSV41_TOKENIZER_CACHE_EPOCH_FILE` at a readable file and replace its
contents between drained runs.
