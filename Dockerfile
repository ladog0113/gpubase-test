FROM ubuntu:22.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENV REPO_URL=https://github.com/ladog0113/gpubase-test \
    REPO_BRANCH=main \
    TARGET_DIR=/home/local/gpubase-test

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
