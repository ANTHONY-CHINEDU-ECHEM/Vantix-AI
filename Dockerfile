# Vantix AI production image.
#   * Dependencies are installed from requirements/runtime.lock with hash verification.
#   * The runtime stage contains no compilers or build tools and runs as an unprivileged user.
#   * The same image serves every role: "serve" (API), "worker" (training) and "migrate".

# Explicit base tags so hadolint can check them and Dependabot can update them.
FROM python:3.12-slim-bookworm AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements/runtime.lock requirements/runtime.lock
RUN pip install --require-hashes --no-deps -r requirements/runtime.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-deps --wheel-dir /wheels . && pip install --no-deps /wheels/*.whl \
    && python -c "import vantix, fastapi, sqlalchemy, psycopg; print('vantix', vantix.__version__)"

FROM python:3.12-slim-bookworm AS runtime
ARG VERSION=0.0.0
ARG REVISION=unknown
ARG CREATED=unknown
LABEL org.opencontainers.image.title="Vantix AI" \
      org.opencontainers.image.description="Hybrid recommender and RAG engine for project management platforms" \
      org.opencontainers.image.vendor="Phexara Ltd" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.created="${CREATED}"
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VANTIX_HOST=0.0.0.0 \
    VANTIX_PORT=8080 \
    VANTIX_DATABASE_URL=sqlite:////data/vantix.db \
    VANTIX_MODEL_DIR=/data/models \
    VANTIX_WORKER_HEARTBEAT_FILE=/tmp/vantix_worker_heartbeat
RUN groupadd --system --gid 10001 vantix \
    && useradd --system --uid 10001 --gid vantix --home-dir /app --shell /usr/sbin/nologin vantix \
    && mkdir -p /data /app \
    && chown -R vantix:vantix /data /app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY --chown=vantix:vantix examples ./examples
USER 10001:10001
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/v1/health', timeout=3).status == 200 else 1)"]
ENTRYPOINT ["vantix"]
CMD ["serve"]
