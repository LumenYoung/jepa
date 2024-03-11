# FROM mambaorg/micromamba:jammy-cuda-12.1.1 as micromamba

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN /usr/local/bin/_dockerfile_setup_root_prefix.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
CMD ["/bin/bash"]

USER root
RUN apt update && apt install -y git curl ffmpeg libsm6 libxext6
# RUN micromamba shell init --shell bash && echo "micromamba activate base" >> /root/.bashrc # make root 

# COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yaml /tmp/env.yaml
# RUN micromamba install -y -n base -f /tmp/env.yaml && \
#   micromamba clean --all --yes


