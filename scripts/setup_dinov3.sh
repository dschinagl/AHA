#!/usr/bin/env bash
set -eu

TARGET_DIR="${1:-dinov3}"
PINNED_COMMIT="31703e4cbf1ccb7c4a72daa1350405f86754b6d1"

if [ -d "$TARGET_DIR" ]; then
  exit 0
fi

git clone https://github.com/facebookresearch/dinov3 "$TARGET_DIR"
git -C "$TARGET_DIR" checkout "$PINNED_COMMIT"

echo "Cloned to $TARGET_DIR."
