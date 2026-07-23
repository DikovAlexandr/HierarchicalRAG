FROM python:3.11.11-slim-bookworm@sha256:a8e0a3090316aed0b11037aac613aef32fb1747dcc1dcb5c0f6c727a0113a07f

ARG SOURCE_REVISION=unversioned

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HIERARCHICAL_RAG_IMAGE_REVISION=${SOURCE_REVISION}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY environments/eval-cpu.lock /tmp/eval-cpu.lock
RUN python -m pip install --no-cache-dir --only-binary=:all: \
    --require-hashes --requirement /tmp/eval-cpu.lock

WORKDIR /opt/hierarchical-rag
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-build-isolation --no-deps .

WORKDIR /workspace
