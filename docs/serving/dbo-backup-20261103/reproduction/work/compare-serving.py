"""Compare matching serving runs without hiding failures or GPU cost."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('baseline', type=Path)
parser.add_argument('candidate', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
baseline = json.loads(args.baseline.read_text())
candidate = json.loads(args.candidate.read_text())
points = {p['concurrency']: p for p in baseline['points']}
rows = []
for optimized in candidate['points']:
    c = optimized['concurrency']
    reference = points[c]
    for sample in reference['samples'] + optimized['samples']:
        assert sample['max_concurrency'] == c
        assert sample['num_prompts'] == max(16, 4*c)
        assert sample['model_id'] == 'deepseek-v4.1-flash-1017'
    old = reference['median_output_tokens_per_second']
    new = optimized['median_output_tokens_per_second']
    rows.append(dict(
        concurrency=c,
        baseline_output_tok_s=old,
        optimized_output_tok_s=new,
        total_throughput_ratio=new/old,
        baseline_output_tok_s_per_d_gpu=old/baseline['decode_gpus'],
        optimized_output_tok_s_per_d_gpu=new/candidate['decode_gpus'],
        per_d_gpu_ratio=(new/candidate['decode_gpus'])/(old/baseline['decode_gpus']),
        baseline_ttft_ms=reference['median_ttft_ms'],
        optimized_ttft_ms=optimized['median_ttft_ms'],
        baseline_tpot_ms=reference['median_tpot_ms'],
        optimized_tpot_ms=optimized['median_tpot_ms'],
        baseline_failures=reference['failed'],
        optimized_failures=optimized['failed'],
        baseline_repetitions=reference['repetitions'],
        optimized_repetitions=optimized['repetitions'],
        both_zero_error_pass=reference['all_zero_error_pass'] and optimized['all_zero_error_pass'],
    ))
args.output.write_text(json.dumps(dict(
    baseline=args.baseline.name,
    candidate=args.candidate.name,
    statistic='Median of per-run mean throughput, TTFT and TPOT; all repetitions retained',
    rows=rows,
), indent=2)+'\n')
print(json.dumps(rows, indent=2))
