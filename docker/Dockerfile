# FROM mambaorg/micromamba:jammy-cuda-12.1.1 as micromamba

FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

ENV TZ=Europe/Berlin

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq git curl ffmpeg libsm6 libxext6 tigervnc-standalone-server tigervnc-xorg-extension tigervnc-viewer

SHELL ["/bin/bash", "-c"]

RUN echo "\n\n\n\n\n\n" | bash <(curl -L micro.mamba.pm/install.sh)

RUN /root/.local/bin/micromamba create -n jepa python=3.9 -y

SHELL ["/root/.local/bin/micromamba", "run", "-n", "jepa", "/bin/bash", "-c"]

RUN echo "export PATH=\$PATH:~/.local/bin" && \
  ~/.local/bin/micromamba shell init --shell bash 

RUN echo "micromamba activate jepa" >> /root/.bashrc

RUN pip3 install --no-input torch numpy

# RUN micromamba shell init --shell bash && echo "micromamba activate base" >> /root/.bashrc # make root 

# COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yaml /tmp/env.yaml
# RUN micromamba install -y -n base -f /tmp/env.yaml && \
#   micromamba clean --all --yes
