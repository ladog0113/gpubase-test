#!/bin/sh
set -e

mkdir -p "$TARGET_DIR"

if [ -d "$TARGET_DIR/.git" ]; then
    echo "[gpubase-test] git pull: $TARGET_DIR"
    git config --global --add safe.directory "$TARGET_DIR"
    git -C "$TARGET_DIR" pull --ff-only origin "$REPO_BRANCH"
elif [ -z "$(ls -A "$TARGET_DIR")" ]; then
    echo "[gpubase-test] git clone: $REPO_URL -> $TARGET_DIR"
    git clone --branch "$REPO_BRANCH" "$REPO_URL" "$TARGET_DIR"
else
    echo "[gpubase-test] $TARGET_DIR 에 git 저장소가 아닌 파일이 있어 clone을 건너뜁니다." >&2
fi

ls -l "$TARGET_DIR"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec sleep infinity
