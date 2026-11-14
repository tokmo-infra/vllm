#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---verify}
case "$MODE" in --verify|--apply) ;; *) exit 2 ;; esac
ip -o addr show | grep -Eq '11\.5\.0\.(92|93)/' || { echo 'Unexpected host' >&2; exit 1; }
ROOT=/data2/bench/eng-xinhao/hisparse-pd-correctness-20261113-ced31f9
mkdir -p "$ROOT/logs"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BEFORE="$ROOT/logs/cpu-governor-$STAMP-before.txt"
for POLICY in /sys/devices/system/cpu/cpufreq/policy*; do
  printf '%s ' "$POLICY"
  cat "$POLICY/scaling_driver" "$POLICY/scaling_governor"
done > "$BEFORE"
if [[ "$MODE" == --apply ]]; then
  for POLICY in /sys/devices/system/cpu/cpufreq/policy*; do
    grep -qw performance "$POLICY/scaling_available_governors"
  done
  sudo -n bash -c 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo performance > "$p/scaling_governor"; done'
fi
for POLICY in /sys/devices/system/cpu/cpufreq/policy*; do
  [[ $(cat "$POLICY/scaling_governor") == performance ]] || { echo "Governor mismatch: $POLICY" >&2; exit 1; }
done
printf 'CPU governor=performance verified; saved %s. This does not assert a physical clock lock.\n' "$BEFORE"
