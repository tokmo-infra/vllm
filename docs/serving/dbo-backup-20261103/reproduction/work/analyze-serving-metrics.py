"""Summarize observed serving load, retaining the warmup/measurement scope."""
import argparse
import json
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
names = (
    "num_requests_running", "num_requests_waiting", "spec_decode_num_drafts_total",
    "spec_decode_num_draft_tokens_total", "spec_decode_num_accepted_tokens_total",
    "generation_tokens_total", "num_preemptions_total",
)
pattern = re.compile(r"^vllm:(" + "|".join(names) + r")(?:\{[^}]*\})? ([\d.eE+-]+)$")
points = {}
with tarfile.open(args.archive) as archive:
    for member in archive:
        if not member.name.endswith("/metrics.jsonl"):
            continue
        observations = defaultdict(list)
        errors = defaultdict(int)
        for line in archive.extractfile(member):
            row = json.loads(line)
            if "metrics" not in row:
                errors[row["name"]] += 1
                continue
            values = defaultdict(float)
            for metric in row["metrics"].splitlines():
                match = pattern.match(metric)
                if match:
                    values[match[1]] += float(match[2])
            observations[row["name"]].append(dict(time=row["time"], values=dict(values)))
        summaries = {}
        for endpoint, snapshots in observations.items():
            summary = dict(samples=len(snapshots), first=snapshots[0]["time"],
                           last=snapshots[-1]["time"], sampling_errors=errors[endpoint])
            for metric in names:
                values = [s["values"][metric] for s in snapshots if metric in s["values"]]
                if not values:
                    continue
                if metric.endswith("_total"):
                    summary[metric] = dict(first=values[0], last=values[-1],
                                           delta=values[-1]-values[0],
                                           reset_observed=any(b<a for a,b in zip(values,values[1:])))
                else:
                    positive = [v for v in values if v > 0]
                    summary[metric] = dict(peak=max(values), mean=statistics.mean(values),
                                          median_nonzero=statistics.median(positive) if positive else 0)
            drafted = summary.get("spec_decode_num_draft_tokens_total", {}).get("delta", 0)
            accepted = summary.get("spec_decode_num_accepted_tokens_total", {}).get("delta", 0)
            if drafted:
                summary["draft_token_acceptance_fraction"] = accepted / drafted
            summaries[endpoint] = summary
        points[Path(member.name).parent.name] = summaries
args.output.write_text(json.dumps(dict(
    source=args.archive.name,
    scope="Whole CLI sample including warmup. Counters summed across all exported engines; GPU busy time is not effective SM computation.",
    points=points,
), indent=2) + "\n")
print(f"Summarized {len(points)} samples into {args.output}")
