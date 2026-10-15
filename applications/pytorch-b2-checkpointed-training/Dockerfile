FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
# CPU wheels keep the image small; swap the index for a CUDA build on GPU nodes.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.4"
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home --uid 10001 trainer && mkdir /scratch && chown trainer /scratch

# Credentials come from the environment at run time (Kubernetes Secret, docker --env-file); none are baked in.
USER 10001
WORKDIR /scratch
ENTRYPOINT ["b2-train"]
CMD ["--help"]
