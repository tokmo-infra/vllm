# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Task-local native metrics extension: session capacity, strict >60s tail.

No global perf installation edits; native successful cohort/minute aggregation,
latency distributions and original exporter are retained verbatim.
"""

from itertools import groupby

MINUTE_NS = 60_000_000_000


def capacity_rows(session_count, finished_rows, bounds_start):
    rows = [(bounds_start, session_count)]
    left = session_count
    for stamp, group in groupby(sorted(int(x) for x in finished_rows)):
        left -= sum(1 for _ in group)
        rows.append((stamp, left))
    return rows


def strict_capacity_end(rows, start, end, target):
    value = None
    under = None
    for stamp, changes in groupby(rows, key=lambda x: x[0]):
        final = None
        for _, final in changes:
            pass
        if stamp <= start:
            value = final
            if value >= target:
                under = None
            elif under is None:
                under = stamp
            continue
        if value is None:
            return None, None, "capacity_not_observed_at_maxtpm_start"
        if under is not None and min(stamp, end) - under > MINUTE_NS:
            return (
                under,
                under + MINUTE_NS + 1,
                "capacity_under_target_strictly_over_60s",
            )
        if stamp >= end:
            break
        value = final
        if value >= target:
            under = None
        elif under is None:
            under = stamp
    if value is None:
        return None, None, "capacity_not_observed_at_maxtpm_start"
    if under is not None and end - under > MINUTE_NS:
        return under, under + MINUTE_NS + 1, "capacity_under_target_strictly_over_60s"
    return end, None, None


def install():
    import perf_benchmark.plugins.metrics.standard.backend as backend_module
    import perf_benchmark.plugins.metrics.standard.maxtpm as module

    original = module.dataset_maxtpm
    if getattr(original, "_qwen_capacity_rule", False):
        return

    def task_maxtpm(backend, fact, bounds, occupancy):
        n = fact.scheduling.dataset_session_count
        if n is None:
            raise ValueError(
                "Finite frozen session count required for task capacity rule"
            )
        finished = backend.db.execute(
            "SELECT finished_ns FROM sessions WHERE trial_id=? AND ordinal<? "
            "AND status IN ('succeeded','failed','cancelled') "
            "AND finished_ns IS NOT NULL ORDER BY finished_ns,ordinal",
            [fact.trial_id, n],
        ).fetchall()
        rows = capacity_rows(n, [x[0] for x in finished], bounds[0])
        return original(backend, fact, bounds, rows)

    task_maxtpm._qwen_capacity_rule = True
    module._concurrency_end = strict_capacity_end
    module.FORMULA_VERSION = module.FORMULA_VERSION + "+task-capacity-strict-gt60-v1"
    module.dataset_maxtpm = task_maxtpm
    backend_module.dataset_maxtpm = task_maxtpm
