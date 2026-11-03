"""Compare identical end-to-end questions; preserve every changed answer."""

import argparse
import json
from pathlib import Path


def read_run(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    keyed = {row["index"]: row for row in rows}
    assert len(keyed) == len(rows), "Duplicate question index"
    summary = json.loads(path.with_suffix(".summary.json").read_text())
    assert summary["total"] == len(rows) == 1319, "Require complete GSM8K test"
    return keyed, summary


parser = argparse.ArgumentParser()
parser.add_argument("baseline", type=Path)
parser.add_argument("candidate", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
baseline, before = read_run(args.baseline)
candidate, after = read_run(args.candidate)
assert before["dataset_sha256"] == after["dataset_sha256"]
for key in ("model", "concurrency", "max_tokens", "limit"):
    assert before["config"][key] == after["config"][key], key
assert baseline.keys() == candidate.keys()
changes = []
for index in sorted(baseline):
    a, b = baseline[index], candidate[index]
    assert a["prompt_sha256"] == b["prompt_sha256"], index
    assert a["gold"] == b["gold"], index
    if (
        a.get("prediction"),
        a.get("correct"),
        a.get("finish_reason"),
        a.get("error"),
    ) != (
        b.get("prediction"),
        b.get("correct"),
        b.get("finish_reason"),
        b.get("error"),
    ):
        fields = ("prediction", "correct", "finish_reason", "error", "usage")
        changes.append(
            {
                "index": index,
                "gold": a["gold"],
                "prompt_sha256": a["prompt_sha256"],
                "baseline": {key: a.get(key) for key in fields},
                "candidate": {key: b.get(key) for key in fields},
            }
        )
delta_pp = (after["correct"] - before["correct"]) / 1319 * 100
result = {
    "baseline": str(args.baseline),
    "candidate": str(args.candidate),
    "baseline_summary": before,
    "candidate_summary": after,
    "same_dataset_prompts_and_parameters": True,
    "accuracy_delta_percentage_points": delta_pp,
    "gate": {
        "maximum_drop_percentage_points": 0.5,
        "zero_errors": after["errors"] == 0,
        "accuracy_pass": delta_pp >= -0.5,
        "passed": after["errors"] == 0 and delta_pp >= -0.5,
    },
    "regressions": [
        i
        for i in sorted(baseline)
        if baseline[i].get("correct") and not candidate[i].get("correct")
    ],
    "improvements": [
        i
        for i in sorted(baseline)
        if not baseline[i].get("correct") and candidate[i].get("correct")
    ],
    "changed_answers": changes,
}
args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
print(
    json.dumps(
        {
            key: result[key]
            for key in (
                "accuracy_delta_percentage_points",
                "gate",
                "regressions",
                "improvements",
            )
        },
        ensure_ascii=False,
    )
)
