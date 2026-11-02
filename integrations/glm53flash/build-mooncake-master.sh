#!/usr/bin/env bash
# Build only the pinned Mooncake Master; leave the Store SDK unchanged.
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 SOURCE_DIR BUILD_DIR RUNTIME_OUT [CMake options...]" >&2
  exit 2
fi
source_dir=$(cd "$1" && pwd)
mkdir -p "$2" "$3/lib"
build_dir=$(cd "$2" && pwd)
runtime_dir=$(cd "$3" && pwd)
shift 3
here=$(cd "$(dirname "$0")" && pwd)

if [ -e "$source_dir/.git" ]; then
  test "$(git -C "$source_dir" rev-parse HEAD)" = \
    719735896c86b56fabec6cf3e825fb2ea640597a
fi
for patch_name in mooncake-occupied-accounting.patch mooncake-fit-pressure.patch; do
  patch_file="$here/patches/$patch_name"
  if git -C "$source_dir" apply --check "$patch_file" 2>/dev/null; then
    git -C "$source_dir" apply "$patch_file"
  else
    git -C "$source_dir" apply --reverse --check "$patch_file"
    echo "$patch_name already applied"
  fi
done

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DWITH_TE=ON -DWITH_STORE=ON -DWITH_STORE_RUST=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
  -DUSE_CUDA=OFF -DUSE_NOF=OFF -DENABLE_DEBUG_SYMBOLS=OFF \
  -DCMAKE_BUILD_TYPE=Release '-DCMAKE_CXX_FLAGS_RELEASE=-O2 -DNDEBUG' "$@"
cmake --build "$build_dir" --target mooncake_master -j"${BUILD_JOBS:-3}"
binary="$build_dir/mooncake-store/src/mooncake_master"
install -m 755 "$binary" "$runtime_dir/mooncake_master"
ldd "$binary" > "$runtime_dir/ldd.txt"
if grep -q 'not found' "$runtime_dir/ldd.txt"; then
  cat "$runtime_dir/ldd.txt" >&2
  exit 1
fi
while IFS= read -r library; do
  case "$(basename "$library")" in
    libc.so.*|libm.so.*|libpthread.so.*|librt.so.*|libdl.so.*|libresolv.so.*) continue ;;
  esac
  cp -L "$library" "$runtime_dir/lib/"
done < <(awk '$2 == "=>" && $3 ~ /^\// {print $3}' "$runtime_dir/ldd.txt")
sha256sum "$runtime_dir/mooncake_master"
