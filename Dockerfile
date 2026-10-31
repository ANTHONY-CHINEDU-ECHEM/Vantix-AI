# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[postgres]"

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VANTIX_HOST=0.0.0.0 \
    VANTIX_PORT=8080 \
    VANTIX_DATABASE_URL=sqlite:////data/vantix.db \
    VANTIX_MODEL_DIR=/data/models
RUN groupadd --system vantix && useradd --system --gid vantix --home /app vantix \
    && mkdir -p /data /app && chown -R vantix:vantix /data /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
WORKDIR /app
COPY --chown=vantix:vantix examples ./examples
USER vantix
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/v1/health', timeout=3).status == 200 else 1)"
CMD ["vantix", "serve"]
