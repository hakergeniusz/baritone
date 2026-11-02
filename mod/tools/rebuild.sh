#!/bin/zsh
# One-command rebuild of the baritone mod and the client it runs in:
# flatten the ported sources, apply the mixins, wire the render stage
# patch, build the client (fast profile) and install it next to the
# stock one. The game must not be running (the binary is swapped under
# it) — the script stops with a message if it is.
#
# Usage: mod/tools/rebuild.sh [--skip-tests]
set -e
cd "$(dirname "$0")/.."

RUST_BIN=/home/hakergeniusz/.local/share/horizon-launcher/rust/1.99.1-x86_64-unknown-linux-gnu/bin
CARGO_HOME=/home/hakergeniusz/.local/share/horizon-launcher/cargo
TARGET_DIR=/home/hakergeniusz/.cache/horizon-baritone-build
TMPDIR=/home/hakergeniusz/.cache/tmp-build
WORKSPACE=/home/hakergeniusz/.local/share/horizon-launcher/native/source/af55a734865da4f1b11836164a979ff06a1f0449
MIXINS=/home/hakergeniusz/.local/share/horizon-launcher/native/tools/3f99a5c60dfa/bin/horizon-mixins
STAGE=/home/hakergeniusz/.cache/baritone-stage
INSTALL=/home/hakergeniusz/Downloads/horizon-linux-x86_64/horizon-client-baritone

export PATH="$RUST_BIN:$PATH" CARGO_HOME TMPDIR CARGO_TARGET_DIR="$TARGET_DIR"

if pgrep -x horizon-client >/dev/null; then
  echo "rebuild: a horizon-client is running — quit it first (the binary is swapped in place)"
  exit 1
fi

echo "== assemble (flatten the ported sources into the mixin payload)"
python3 tools/assemble.py

if [ "$1" != "--skip-tests" ]; then
  echo "== mod tests"
  cargo test --locked --lib
fi

echo "== apply mixins"
trash "$STAGE" 2>/dev/null || true
"$MIXINS" apply --workspace "$WORKSPACE" --mods "$PWD/mixins" --out "$STAGE" | rg 'stage_changed'

echo "== stage patch (render hook)"
python3 tools/stage_patch.py "$STAGE"

echo "== build client (fast profile)"
(cd "$STAGE" && cargo build --profile fast --locked -p horizon-client)

echo "== install"
cp "$TARGET_DIR/fast/horizon-client" "$INSTALL"
echo "rebuild: installed $INSTALL"
