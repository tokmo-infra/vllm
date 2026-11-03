"""Read GPU counters and raw RDMA port counters without submitting GPU work."""
import json
import subprocess
import time
from pathlib import Path

fields = "index,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,temperature.gpu"
started = time.time()
result = subprocess.run(
    ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
    text=True, capture_output=True, timeout=5,
)
gpu = [dict(zip(fields.split(","), map(str.strip, line.split(","))))
       for line in result.stdout.splitlines()]
counters = {}
for index in (0, 6, 8, 10, 12, 14, 18, 20):
    hca = f"mlx5_{index}"
    root = Path(f"/sys/class/infiniband/{hca}/ports/1/counters")
    counters[hca] = {}
    for name in ("port_xmit_data", "port_rcv_data", "port_xmit_packets",
                 "port_rcv_packets", "port_xmit_wait", "port_xmit_discards",
                 "port_rcv_errors", "link_downed"):
        try:
            counters[hca][name] = int((root / name).read_text())
        except (OSError, ValueError) as exc:
            counters[hca][name] = {"error": str(exc)}
print(json.dumps(dict(time=started, finished=time.time(), gpu=gpu,
                     gpu_exit_code=result.returncode, gpu_error=result.stderr,
                     rdma_raw_counters=counters)))
