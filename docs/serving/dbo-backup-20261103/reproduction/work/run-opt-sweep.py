import json,subprocess,sys
from pathlib import Path
root=Path('/work/performance/optimized-dep16-p5d5-v7-001')
for c in (1,8,32,64,128,256,512,1024,2048,4096):
 for r in range(1,4):
  subprocess.run([sys.executable,'/work/run-opt-bench.py','--ep','16','--concurrency',str(c),f'--run-tag=-r{r}'],check=True)
  p=root/f'ep16-c{c}-r{r}'
  j=json.loads((p/'result.json').read_text())
  a={k:j.get(k) for k in ('completed','failed','output_throughput')}
  a.update(expected=max(16,c*4),zero_error_pass=j['completed']==max(16,c*4) and j['failed']==0)
  (p/'acceptance.json').write_text(json.dumps(a,indent=2))
  print(p.name,a,flush=True)
