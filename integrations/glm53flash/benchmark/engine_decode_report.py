"""Per-request engine decode timing; never substitutes client SSE timestamps."""
from collections import Counter
import json
import math
import sqlite3

from cohort50 import distribution


def summarize(rows):
    speeds, tpots, missing = [], [], Counter()
    for row in rows:
        try:
            tokens = json.loads(row["usage_json"])["completion_tokens"]
            timing = json.loads(row["extension_json"]).get("server_metrics")
            if not timing:
                missing["engine_timing_not_captured"] += 1
                continue
            ms = timing.get("generation_time_ms")
            if type(tokens) is not int or tokens <= 1:
                missing["insufficient_output_tokens"] += 1
            elif type(ms) not in (int, float) or not math.isfinite(ms) or ms <= 0:
                missing["engine_decode_interval_unavailable"] += 1
            else:
                tpots.append(ms / (tokens - 1))
                speeds.append((tokens - 1) * 1000 / ms)
        except (KeyError, TypeError, ValueError):
            missing["invalid_engine_timing_or_usage"] += 1
    return {
        "source": "vllm_engine_core_first_to_last_generated_token",
        "formula": "(completion_tokens-1)*1000/generation_time_ms",
        "aggregation": "arithmetic_mean_and_quantile_disc; no weighting_or_clipping",
        "decode_tokens_per_second": distribution(speeds, missing, len(rows)),
        "tpot_ms": distribution(tpots, missing, len(rows)),
    }


def build_report(database, native):
    if "start_ns" not in native or "window_end_ns" not in native:
        return {"availability": "unavailable", "reason": "no_verified_cohort_window"}
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            """SELECT f.usage_json,f.extension_json FROM turns t
            JOIN turn_keys k ON k.id=t.id JOIN sessions s ON s.id=k.session_key
            JOIN request_facts f ON f.id=t.fact_id
            WHERE t.status='succeeded' AND s.trial_id=?
              AND f.response_completed_ns>=? AND f.response_completed_ns<?""",
            (native["trial"]["trial_id"], native["start_ns"], native["window_end_ns"]),
        ).fetchall()
    report = summarize(rows)
    report["cohort"] = {key: native[key] for key in ("start_ns", "window_end_ns", "complete_minutes")}
    return report
