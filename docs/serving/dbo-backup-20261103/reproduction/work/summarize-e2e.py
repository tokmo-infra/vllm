"""Keep every repetition and failure when summarizing serving measurements."""
import argparse
import hashlib
import json
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('archive', type=Path)
p.add_argument('output', type=Path)
p.add_argument('--decode-gpus', type=int, required=True)
args = p.parse_args()
points = defaultdict(list)
with tarfile.open(args.archive) as archive:
    for member in archive:
        if not member.name.endswith('/result.json'):
            continue
        match = re.search(r'/ep(\d+)-c(\d+)-r(\d+)/result.json$', member.name)
        if not match:
            continue
        ep, concurrency, repetition = map(int, match.groups())
        with archive.extractfile(member) as handle:
            source = json.load(handle)
        # The complete per-request arrays remain in the immutable raw archive.
        result = {k: v for k, v in source.items() if not isinstance(v, (list, dict))}
        result.update(ep=ep, concurrency=concurrency, repetition=repetition)
        result['expected_requests'] = max(16, concurrency * 4)
        result['zero_error_pass'] = (
            result['completed'] == result['expected_requests']
            and result['failed'] == 0
        )
        points[concurrency].append(result)

output = dict(
    raw_archive=args.archive.name,
    raw_archive_sha256=hashlib.file_digest(args.archive.open('rb'), 'sha256').hexdigest(),
    decode_gpus=args.decode_gpus,
    points=[],
)
for c, samples in sorted(points.items()):
    samples.sort(key=lambda s: s['repetition'])
    rates = [s['output_throughput'] for s in samples]
    output['points'].append(dict(
        concurrency=c, repetitions=len(samples),
        completed=sum(s['completed'] for s in samples),
        failed=sum(s['failed'] for s in samples),
        all_zero_error_pass=all(s['zero_error_pass'] for s in samples),
        median_output_tokens_per_second=statistics.median(rates),
        min_output_tokens_per_second=min(rates),
        max_output_tokens_per_second=max(rates),
        median_output_tokens_per_second_per_decode_gpu=statistics.median(rates)/args.decode_gpus,
        median_ttft_ms=statistics.median(s['mean_ttft_ms'] for s in samples),
        median_tpot_ms=statistics.median(s['mean_tpot_ms'] for s in samples),
        samples=samples,
    ))
args.output.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps([{k: v for k, v in point.items() if k != 'samples'} for point in output['points']], indent=2))
