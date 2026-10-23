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
    # 새 볼륨에 lost+found 등이 있어 비어 있지 않은 경우: 기존 파일은 두고 저장소 파일만 받는다
    echo "[gpubase-test] git init + fetch: $REPO_URL -> $TARGET_DIR"
    git config --global --add safe.directory "$TARGET_DIR"
    git -C "$TARGET_DIR" init -q
    git -C "$TARGET_DIR" remote add origin "$REPO_URL"
    git -C "$TARGET_DIR" fetch -q origin "$REPO_BRANCH"
    git -C "$TARGET_DIR" checkout -q -f -B "$REPO_BRANCH" FETCH_HEAD
fi

ls -l "$TARGET_DIR"

if [ "$#" -gt 0 ]; then
    exec "$@"
fi

exec sleep infinity
