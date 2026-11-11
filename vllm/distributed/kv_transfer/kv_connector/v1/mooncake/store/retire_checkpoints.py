# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retire rank-local recurrent tails after a complete successor is persisted."""

import logging
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)
RETIRE_DELAY_S = 10.0


def retirement_enabled(config):
    enabled = config.get("remove_superseded_checkpoints", False)
    if type(enabled) is not bool:
        raise ValueError("remove_superseded_checkpoints must be a boolean")
    return enabled


def validate_retirement_scope(config, tp, pcp, dcp, pp, load_failure_policy="fail"):
    enabled = retirement_enabled(config)
    if enabled and (
        (tp, pcp, dcp, pp) != (1, 1, 1, 1)
        or str(config.get("enable_group_semantics", False)).strip().lower() == "true"
        or config.get("store_tp_size") is not None
        or config.get("enable_store_tp_lcm")
    ):
        raise ValueError(
            "checkpoint retirement requires rank-local TP/PCP/DCP/PP=1 "
            "without group semantics or Store-TP remapping"
        )
    return enabled


def mark_prompt_retirement(scheduler, tracker, meta, checkpoint_position):
    """Arm only the actual prompt-end cohort, never shared/decode checkpoints."""
    if (
        not getattr(scheduler, "remove_superseded_checkpoints", False)
        or tracker is None
    ):
        return
    if scheduler.kv_role == "kv_consumer" or not meta.can_save:
        return
    coord = scheduler._store_coord
    prompt = meta.num_prompt_tokens
    if prompt is None or prompt != tracker.prefill_end_tokens:
        return  # No retirement on resumed/replayed generations.
    new = checkpoint_position(prompt, coord.hash_block_size, coord.use_eagle)
    old = tracker.admitted_tokens
    if (
        not 0 < old < new
        or old % coord.hash_block_size
        or old % scheduler._block_size == 0
        or new % scheduler._block_size == 0
        or meta.completed_token_len is None
        or meta.completed_token_len < new
    ):
        return
    groups = coord.mamba_group_ids
    handed = {g for g, block, n in meta.boundary_puts or [] if n == new and block}
    if not groups or not groups <= handed:
        return
    meta.superseded_boundary = old
    meta.successor_boundary = new


@dataclass(frozen=True)
class RetirementPlan:
    old_keys: tuple[str, ...]
    new_keys: tuple[str, ...]


class CheckpointRetirer:
    def __init__(self, sender):
        if not callable(getattr(sender.store, "batch_remove", None)):
            raise ValueError("native Store must expose batch_remove(keys, force)")
        self.sender = sender
        self.pending: deque[tuple[float, RetirementPlan]] = deque()

    def _plan(self, meta):
        sender = self.sender
        if sender.enable_group_semantics:
            return None
        old = meta.superseded_boundary
        new = meta.successor_boundary
        hbs = sender.coord.hash_block_size
        if (
            old is None
            or new is None
            or not 0 < old < new
            or old % hbs
            or new % hbs
            or old % sender.block_size == 0
            or new % sender.block_size == 0
            or new // hbs > len(meta.block_hashes)
        ):
            return None
        old_keys, new_keys = [], []
        for group in sorted(sender.coord.mamba_group_ids):
            db = sender.token_databases[group]
            if (
                not sender.group_participates[group]
                or sender.group_put_steps[group] != 1
                or tuple(db.store_layout.local_shard_ids) != (0,)
            ):
                return None  # All necessary shards must be local and checked.
            blocks = {
                block
                for g, block, n in meta.boundary_puts or []
                if g == group and n == new and block
            }
            if len(blocks) != 1:
                return None  # Missing/conflicting exact successor source.
            new_keys.append(db.key_for(meta.block_hashes[new // hbs - 1]))
            if old % db.block_size:
                old_keys.append(db.key_for(meta.block_hashes[old // hbs - 1]))
        # Include the exact Attention gap/proof objects needed by this tail.
        # A successful Mamba-only batch is not a complete prompt-tail cohort.
        start = new // sender.block_size * sender.block_size
        for group, db in enumerate(sender.token_databases):
            if group in sender.coord.mamba_group_ids:
                continue
            if (
                not sender.group_participates[group]
                or sender.group_put_steps[group] != 1
                or tuple(db.store_layout.local_shard_ids) != (0,)
            ):
                return None
            proof = new + sender.coord.eagle_proof_margin_by_group.get(group, 0)
            if (
                meta.completed_token_len is None
                or proof > meta.completed_token_len
                or proof % hbs
                or proof // hbs > len(meta.block_hashes)
            ):
                return None
            for block in range(
                start // db.block_size, (proof + db.block_size - 1) // db.block_size
            ):
                end = min((block + 1) * db.block_size, proof)
                sources = {
                    bid
                    for g, bid, n in meta.boundary_puts or []
                    if g == group and n == end and bid
                }
                if len(sources) != 1:
                    return None
                new_keys.append(db.key_for(meta.block_hashes[end // hbs - 1]))
        if not old_keys or not new_keys or set(old_keys) & set(new_keys):
            return None
        return RetirementPlan(tuple(old_keys), tuple(new_keys))

    def _successor_exists(self, plan):
        try:
            result = list(self.sender.store.batch_is_exist(list(plan.new_keys)))
            return len(result) == len(plan.new_keys) and all(x == 1 for x in result)
        except Exception:
            logger.exception("checkpoint retirement successor verification failed")
            return False

    def after_persist(self, meta):
        """Call only after a successful boundary PUT/exists batch."""
        plan = self._plan(meta)
        if plan is None or not self._successor_exists(plan):
            return
        sender = self.sender
        with sender.done_task_lock:
            if meta.store_job_id not in sender.stored_requests.get(meta.req_id, ()):
                return
            self.pending.append((time.monotonic() + RETIRE_DELAY_S, plan))

    def collect_due(self):
        """Sending thread only; no timer, forced deletion, or leased-key retry."""
        now = time.monotonic()
        while self.pending and self.pending[0][0] <= now:
            _, plan = self.pending.popleft()
            # The successor may have been evicted since persistence. Recheck
            # the entire cohort before deletion; never renew the OLD lease.
            if not self._successor_exists(plan):
                continue
            try:
                result = list(
                    self.sender.store.batch_remove(list(plan.old_keys), False)
                )
                if len(result) != len(plan.old_keys):
                    logger.error("checkpoint retirement returned malformed statuses")
                    continue
                logger.info(
                    "checkpoint retirement attempted=%d removed=%d refused=%d",
                    len(result),
                    result.count(0),
                    sum(x != 0 for x in result),
                )
            except Exception:
                logger.exception("checkpoint retirement failed; not retried")
