#!/usr/bin/env bash
# Build the Python runtime that ships inside the .app.
#
#   ./desktop/build_runtime.sh
#
# Produces desktop/src-tauri/runtime/, which tauri.conf.json copies into
# Contents/Resources. After this the app needs no checkout, no uv, and no venv:
# it is the thing "self-contained" was supposed to mean.
#
# WHY A COPIED DISTRIBUTION AND NOT A VENV
#
# A venv symlinks its interpreter back to the base Python it was created from,
# and records that base as an absolute path in pyvenv.cfg. Move the venv to
# another machine and it points at a directory that is not there. `uv venv
# --relocatable` fixes the *scripts*, not the interpreter, so it does not solve
# this on its own.
#
# python-build-standalone distributions -- which is what `uv python install`
# fetches -- are built to be relocated. Copying one and installing straight into
# its site-packages gives a tree that works wherever it lands.
#
# Console scripts still bake an absolute shebang at install time, so nothing
# here relies on them: the app runs `bin/python3.12 -m voiceagent.web.server`.
#
# WHAT IS NOT IN HERE
#
# Model weights. Together they are about 7.6 GB against the runtime's ~1.2 GB,
# and a 9 GB installer is not a thing people download. They are fetched on first
# use into the usual Hugging Face cache, which is how every local-AI app does it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"
RUNTIME="$HERE/src-tauri/runtime"
PYTHON_VERSION="3.12"

echo "==> locating a standalone Python $PYTHON_VERSION"
uv python install "$PYTHON_VERSION" >/dev/null 2>&1 || true

# NOT `uv python find`. It prefers a venv discovered in the working directory,
# so from inside this project it returns .venv -- and copying a venv is exactly
# the mistake this script exists to prevent. This build made it once: the copy
# passed every check locally, because its symlinked interpreter still resolved
# on the machine that built it. Ask for the installed list instead and take the
# uv-managed distribution, which is the relocatable one.
BASE="$(uv python list --only-installed --output-format json | python3 -c '
import json, sys
for entry in json.load(sys.stdin):
    path = entry.get("path") or ""
    if "/uv/python/" in path and (entry.get("version") or "").startswith(sys.argv[1]):
        print(path)
        break
' "$PYTHON_VERSION")"

test -n "$BASE" || {
    echo "no uv-managed Python $PYTHON_VERSION found."
    echo "run: uv python install $PYTHON_VERSION"
    exit 1
}
BASE_ROOT="$(cd "$(dirname "$BASE")/.." && pwd -P)"   # -P: resolve the version symlink
test ! -f "$BASE_ROOT/pyvenv.cfg" || { echo "$BASE_ROOT is a venv, not a distribution"; exit 1; }
echo "    $BASE_ROOT"

echo "==> copying the distribution into the bundle"
rm -rf "$RUNTIME"
mkdir -p "$(dirname "$RUNTIME")"
cp -R "$BASE_ROOT" "$RUNTIME"
chmod -R u+w "$RUNTIME"

RUNTIME_PY="$RUNTIME/bin/python$PYTHON_VERSION"
test -x "$RUNTIME_PY" || { echo "no interpreter at $RUNTIME_PY"; exit 1; }

# Assert we copied a distribution, not a venv. A venv copy fails only once the
# .app is on someone else's Mac, which is the worst place to discover it.
test ! -f "$RUNTIME/pyvenv.cfg" || {
    echo "runtime contains pyvenv.cfg: a venv was copied instead of a distribution"
    exit 1
}
if [ -L "$RUNTIME_PY" ]; then
    LINK_TARGET="$(cd "$RUNTIME/bin" && readlink "python$PYTHON_VERSION")"
    case "$LINK_TARGET" in
        /*) echo "interpreter symlinks outside the bundle: $LINK_TARGET"; exit 1 ;;
    esac
fi

# uv marks its own managed installs EXTERNALLY-MANAGED so nothing modifies them
# in place. That guard is about uv's copy, not this one -- the whole point here
# is a private distribution we are allowed to install into.
find "$RUNTIME" -name "EXTERNALLY-MANAGED" -delete

# python-build-standalone bakes its build-time prefix into the binary and only
# overrides it if PYTHONHOME says otherwise. Without this the copied interpreter
# reports sys.prefix back at the original uv directory and imports from there --
# so the bundle looks fine on the build machine and has no packages anywhere
# else. Everything below, and the launcher in main.rs, sets it.
export PYTHONHOME="$RUNTIME"

echo "==> installing the project and its extras"
# --link-mode=copy so nothing in the bundle is a hardlink into uv's cache; a
# hardlinked bundle looks fine locally and ships broken.
uv pip install \
    --python "$RUNTIME_PY" \
    --link-mode=copy \
    --no-cache \
    "$PROJECT[stt,llm,tts,clone,loop,tools,indic]" >/dev/null

echo "==> trimming"
# Caches and bytecode only. Not source: an install that differs from the tested
# one is a bad trade for a few megabytes.
find "$RUNTIME" -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find "$RUNTIME" -name "*.pyc" -delete 2>/dev/null || true

echo "==> verifying it runs detached from this checkout"
# From / with an empty environment. If anything still reaches back into the
# project directory or the developer's PATH, it fails here and not on a user's
# machine.
( cd / && env -i HOME="$HOME" PYTHONHOME="$RUNTIME" "$RUNTIME_PY" - <<CHECK
import sys

runtime = "$RUNTIME"
assert sys.prefix == runtime, f"prefix is {sys.prefix}, not the bundle"
assert all("/src" not in p or runtime in p for p in sys.path), \
    f"a checkout is on sys.path: {sys.path}"

import voiceagent
import voiceagent.web.server  # noqa: F401
from voiceagent.text.num2words_shim import install, is_installed

assert voiceagent.__file__.startswith(runtime), \
    f"imported from outside the bundle: {voiceagent.__file__}"
install()
assert is_installed()
import num2words

assert num2words.num2words(1999, to="year") == "nineteen ninety-nine"
print(f"    prefix {sys.prefix}")
print("    imports clean, shim active, no checkout on sys.path")
CHECK
)

echo "==> size"
du -sh "$RUNTIME"
echo "done: $RUNTIME"
