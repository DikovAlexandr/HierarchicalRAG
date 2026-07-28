FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime@sha256:831247999fbf7e08f61b3e39f6d77ee434f38f6f07f769d00db451e853878067

ARG SOURCE_REVISION=unversioned

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HIERARCHICAL_RAG_IMAGE_REVISION=${SOURCE_REVISION}

COPY environments/hrm-text-gpu.lock /tmp/hrm-text-gpu.lock
RUN python -m pip install --no-cache-dir --no-deps --require-hashes \
    --requirement /tmp/hrm-text-gpu.lock

COPY pyproject.toml README.md /opt/hierarchical-rag/
COPY src /opt/hierarchical-rag/src
WORKDIR /opt/hierarchical-rag
RUN python -m pip install --no-build-isolation --no-deps .

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "hierarchical_rag.run_dense_calibration"]
