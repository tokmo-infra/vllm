#!/bin/sh
set -eu
root=/opt/glm53flash-mooncake-master
export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$root/mooncake_master" "$@"
