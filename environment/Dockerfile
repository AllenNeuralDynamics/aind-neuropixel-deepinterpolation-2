# hash:sha256:4f762dd94c8d5e6f78729393c7e2d76c08c91c618f2d07b6a6c913ce84c8b715
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    numcodecs[wavpack]==0.12.1 \
    numpy==1.24.4 \
    torch==2.4.1 \
    wavpack-numcodecs==0.2.2 \
    zarr==2.16.1
