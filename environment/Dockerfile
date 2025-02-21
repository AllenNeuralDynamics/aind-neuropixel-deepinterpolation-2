# hash:sha256:f613993332a13ef39a27091b6e8e9dbb13ffbafba8f1b7cd0888a295c8d6c2b1
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        htop \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    numcodecs[wavpack]==0.12.1 \
    numpy==1.24.4 \
    torch==2.4.1 \
    wavpack-numcodecs==0.2.2 \
    zarr==2.16.1
