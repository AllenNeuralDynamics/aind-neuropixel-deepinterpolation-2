# hash:sha256:8fe800193a140b1c6bcf3b0ccf371f19800cfd37fd251151a524285f98938c10
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch-tensorflow-jax:2.1

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        htop=3.0.5-7build2 \
        screen=4.9.0-1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    numcodecs[wavpack]==0.12.1 \
    numpy==1.24.4 \
    wavpack-numcodecs==0.2.2 \
    zarr==2.16.1
