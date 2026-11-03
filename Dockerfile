FROM gcc:14-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash coreutils findutils procps util-linux \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 grader \
    && useradd --uid 1000 --gid 1000 --create-home --shell /usr/sbin/nologin grader

WORKDIR /work
USER 1000:1000
CMD ["sh", "-lc", "sleep infinity"]
