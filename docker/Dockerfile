FROM mambaorg/micromamba:jammy-cuda-12.1.1
# FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 

ENV TZ=Europe/Berlin
COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yaml /tmp/env.yaml

USER root

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \ 
  DEBIAN_FRONTEND=noninteractive apt-get install -yq \
  git curl ffmpeg libsm6 libxext6 \
  tigervnc-standalone-server tigervnc-xorg-extension gcc g++ wget xfce4 xfce4-goodies \
  && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

RUN micromamba install -y -n base -f /tmp/env.yaml && \
  micromamba clean --all --yes

WORKDIR /repo

# RUN micromamba shell init --shell bash && echo "micromamba activate base" >> /root/.bashrc # make root 
