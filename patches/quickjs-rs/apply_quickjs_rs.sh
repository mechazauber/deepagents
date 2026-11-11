#!/usr/bin/env bash
# Apply the quickjs-rs memfd + wasmtime thread-affinity patches to the
# installed quickjs_rs site-packages (the dcode uv-tool environment).
#
# quickjs-rs is a SEPARATE PyPI package (not part of the deepagents source
# tree), so these patches target the installed copy:
#   <uv tool env>/lib/python3.14/site-packages/quickjs_rs/
#
# Two patches:
#   1. _wasmtime.py  — set Config.memory_init_cow = False so the shared
#      wasmtime engine does not back linear-memory init with a memfd
#      (memfd_create), which fails in sandboxes that block memfd (EPERM).
#   2. context.py    — wasmtime thread affinity: the engine's Store is NOT
#      thread-safe; pin every store access onto a single affinity thread so
#      the `task()` bridge (which hops to the parent LangGraph loop via
#      run_coroutine_threadsafe) cannot corrupt the store cross-thread.
#
# Usage:
#   bash apply_quickjs_rs.sh [SITE_PACKAGES_DIR]
#
# If SITE_PACKAGES_DIR is omitted, it auto-detects the dcode uv-tool env.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect the dcode uv-tool site-packages if not given.
if [[ $# -ge 1 ]]; then
  SITE="$1"
else
  SITE="$(python3 -c 'import quickjs_rs, os; print(os.path.dirname(quickjs_rs.__file__))' 2>/dev/null || true)"
  if [[ -z "$SITE" ]]; then
    SITE="$(ls -d /sandbox/.local/share/uv/tools/deepagents-code/lib/python*/site-packages/quickjs_rs 2>/dev/null | head -1)"
  fi
  if [[ -z "$SITE" ]]; then
    echo "ERROR: could not locate the installed quickjs_rs package." >&2
    echo "Pass the site-packages dir explicitly: bash apply_quickjs_rs.sh /path/to/site-packages/quickjs_rs" >&2
    exit 1
  fi
fi

echo "Target quickjs_rs: $SITE"

apply_one() {
  local patch="$1"
  local label="$2"
  # The diff was generated as:  --- <orig>  +++ <installed>
  # `patch` needs the target file to match the '---' (original) side.
  # Re-run is idempotent-safe: if already applied, patch reports "Reversed
  (or previously applied) patch detected" and we skip.
  if patch -N -r - "$SITE/$(basename "$patch" | sed 's/\.patch$//; s/^quickjs_rs_//')" < "$patch" >/dev/null 2>&1; then
    echo "  [applied] $label"
  elif patch -N -r - "$SITE/$(basename "$patch" | sed 's/\.patch$//; s/^quickjs_rs_//')" < "$patch" >/dev/null 2>&1; then
    echo "  [applied] $label"
  else
    # Already applied (reversed detected) -> treat as OK.
    if patch -N -R --dry-run "$SITE/$(basename "$patch" | sed 's/\.patch$//; s/^quickjs_rs_//')" < "$patch" >/dev/null 2>&1; then
      echo "  [already applied] $label"
    else
      echo "  [FAILED] $label — manual review needed" >&2
      return 1
    fi
  fi
}

echo "Applying quickjs-rs patches..."
apply_one "$SCRIPT_DIR/quickjs_rs_wasmtime_memfd.patch" "_wasmtime.py (memfd fix)"
apply_one "$SCRIPT_DIR/quickjs_rs_context_thread_affinity.patch" "context.py (thread affinity)"

echo
echo "Verifying patches took effect..."
grep -q "memory_init_cow = False" "$SITE/_wasmtime.py" && echo "  [ok] _wasmtime.py: memory_init_cow = False present" || { echo "  [MISSING] _wasmtime.py memfd fix" >&2; exit 1; }
grep -q "_on_wasmtime_thread" "$SITE/context.py" && echo "  [ok] context.py: _on_wasmtime_thread present" || { echo "  [MISSING] context.py thread affinity" >&2; exit 1; }

echo
echo "Done. The installed quickjs_rs now has the memfd + thread-affinity fixes."
echo "NOTE: a `uv tool install -U deepagents-code` will re-extract the stock"
echo "quickjs-rs and wipe these patches — re-run this script after any upgrade."
