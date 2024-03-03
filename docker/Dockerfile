FROM mambaorg/micromamba:jammy-cuda-12.1.1

USER root
RUN apt update && apt install -y git curl
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
  micromamba clean --all --yes


