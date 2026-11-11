#!/usr/bin/env bash
# Apply the quickjs_rs Wasmtime cross-thread crash fix to a dcode install.
#
# Usage:
#   bash apply_fix.sh                          # auto-detect the uv tool install
#   bash apply_fix.sh /path/to/site-packages   # explicit target
#
# The fix patches quickjs_rs/_engine.py so the 4 host-function trampolines
# no longer crash when a `js_eval` script calls `task()` (subagent) on a
# different thread.

set -euo pipefail

TARGET="${1:-/sandbox/.local/share/uv/tools/deepagents-code/lib/python3.14/site-packages}"
ENGINE="${TARGET}/quickjs_rs/_engine.py"

if [[ ! -f "$ENGINE" ]]; then
  echo "ERROR: $ENGINE not found" >&2
  echo "Usage: bash apply_fix.sh /path/to/site-packages" >&2
  exit 1
fi

# Back up the original
cp "$ENGINE" "${ENGINE}.bak"

# Apply the fix: replace the 4 trampoline bodies
python3 - "$ENGINE" <<'PY'
import re, sys

path = sys.argv[1]
with open(path) as f:
    src = f.read()

# 1. _tramp_host_call
old_call = '''def _tramp_host_call(name_ptr: int, name_len: int, this: int, argc: int, argv_ptr: int) -> int:
    return cast("_Instance", _CUR.inst)._host_call(name_ptr, name_len, this, argc, argv_ptr)'''
new_call = '''def _tramp_host_call(name_ptr: int, name_len: int, this: int, argc: int, argv_ptr: int) -> int:
    inst = getattr(_CUR, "inst", None)
    if inst is None:
        return 0
    return inst._host_call(name_ptr, name_len, this, argc, argv_ptr)'''
assert old_call in src, "Could not find _tramp_host_call (already patched?)"
src = src.replace(old_call, new_call)

# 2. _tramp_host_interrupt
old_int = '''def _tramp_host_interrupt() -> int:
    return cast("_Instance", _CUR.inst)._host_interrupt()'''
new_int = '''def _tramp_host_interrupt() -> int:
    inst = getattr(_CUR, "inst", None)
    if inst is None:
        return 0
    return inst._host_interrupt()'''
assert old_int in src, "Could not find _tramp_host_interrupt (already patched?)"
src = src.replace(old_int, new_int)

# 3. _tramp_host_module_normalize
old_norm = '''def _tramp_host_module_normalize(
    base_ptr: int, base_len: int, spec_ptr: int, spec_len: int, out_len_ptr: int
) -> int:
    return cast("_Instance", _CUR.inst)._host_module_normalize(
        base_ptr, base_len, spec_ptr, spec_len, out_len_ptr
    )'''
new_norm = '''def _tramp_host_module_normalize(
    base_ptr: int, base_len: int, spec_ptr: int, spec_len: int, out_len_ptr: int
) -> int:
    inst = getattr(_CUR, "inst", None)
    if inst is None:
        return 0
    return inst._host_module_normalize(
        base_ptr, base_len, spec_ptr, spec_len, out_len_ptr
    )'''
assert old_norm in src, "Could not find _tramp_host_module_normalize (already patched?)"
src = src.replace(old_norm, new_norm)

# 4. _tramp_host_module_load
old_load = '''def _tramp_host_module_load(name_ptr: int, name_len: int, out_len_ptr: int) -> int:
    return cast("_Instance", _CUR.inst)._host_module_load(name_ptr, name_len, out_len_ptr)'''
new_load = '''def _tramp_host_module_load(name_ptr: int, name_len: int, out_len_ptr: int) -> int:
    inst = getattr(_CUR, "inst", None)
    if inst is None:
        return 0
    return inst._host_module_load(name_ptr, name_len, out_len_ptr)'''
assert old_load in src, "Could not find _tramp_host_module_load (already patched?)"
src = src.replace(old_load, new_load)

with open(path, "w") as f:
    f.write(src)

print(f"Applied fix to {path}")
print("  - _tramp_host_call: null-check added")
print("  - _tramp_host_interrupt: null-check added")
print("  - _tramp_host_module_normalize: null-check added")
print("  - _tramp_host_module_load: null-check added")
PY

# Verify the fix
echo
echo "Verifying _engine.py fix..."
grep -c "getattr(_CUR" "$ENGINE" | xargs -I{} echo "  {} null-checks found (expected 4)"

# --- Part 2: context.py thread affinity ---
CONTEXT="${TARGET}/quickjs_rs/context.py"
if [[ ! -f "$CONTEXT" ]]; then
  echo "WARNING: $CONTEXT not found — skipping context.py fix" >&2
  exit 0
fi

# Check if already patched
if grep -q "_on_wasmtime_thread" "$CONTEXT" 2>/dev/null; then
  echo "  context.py already patched (found _on_wasmtime_thread)"
else
  cp "$CONTEXT" "${CONTEXT}.bak"
  echo "Applying context.py thread-affinity fix..."
  # Apply the unified diff (strip the header, use patch -p1)
  PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PATCH_FILE="${PATCH_DIR}/fix-quickjs-wasmtime-thread-affinity.patch"
  if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: $PATCH_FILE not found" >&2
    exit 1
  fi
  # Extract the diff portion (skip the # header lines)
  python3 - "$PATCH_FILE" "$CONTEXT" <<'PY2'
import difflib, sys

patch_path, target_path = sys.argv[1], sys.argv[2]
with open(patch_path) as f:
    content = f.read()
idx = content.find('--- a/quickjs_rs/context.py')
if idx == -1:
    print("ERROR: could not find diff header in patch")
    sys.exit(1)
diff_text = content[idx:]

with open(target_path) as f:
    original = f.readlines()

# Apply the unified diff
import subprocess
result = subprocess.run(
    ['patch', '-p1', '--batch', '-o', target_path],
    input=diff_text,
    capture_output=True,
    text=True,
    cwd=target_path.rsplit('/', 1)[0],
)
# patch -p1 with -o writes to stdout; redirect properly
# Actually use patch with the file in place:
result = subprocess.run(
    ['patch', '-p1', '--batch'],
    input=diff_text,
    capture_output=True,
    text=True,
    cwd=target_path.rsplit('/', 1)[0],
)
if result.returncode != 0:
    print(f"patch failed: {result.stderr}")
    sys.exit(1)
print(f"  context.py patched: {result.stdout.strip()}")
PY2
  if [[ $? -ne 0 ]]; then
    echo "ERROR: context.py patch failed" >&2
    exit 1
  fi
fi

echo
echo "Verifying context.py fix..."
grep -c "_on_wasmtime_thread" "$CONTEXT" | xargs -I{} echo "  {} _on_wasmtime_thread references found (expected >=10)"

echo
echo "Done. Restart dcode to pick up both fixes."
