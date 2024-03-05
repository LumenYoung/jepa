FROM mambaorg/micromamba:jammy-cuda-12.2.2 as micromamba
# FROM mambaorg/micromamba:jammy-cuda-12.1.1 as micromamba

FROM nvidia/cuda:12.3.2-runtime-ubuntu22.04

ENV MAMBA_ROOT_PREFIX="/opt/conda"
ENV MAMBA_EXE="/bin/micromamba"

COPY --from=micromamba "$MAMBA_EXE" "$MAMBA_EXE"
COPY --from=micromamba /usr/local/bin/_activate_current_env.sh /usr/local/bin/_activate_current_env.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_shell.sh /usr/local/bin/_dockerfile_shell.sh
COPY --from=micromamba /usr/local/bin/_entrypoint.sh /usr/local/bin/_entrypoint.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_initialize_user_accounts.sh /usr/local/bin/_dockerfile_initialize_user_accounts.sh
COPY --from=micromamba /usr/local/bin/_dockerfile_setup_root_prefix.sh /usr/local/bin/_dockerfile_setup_root_prefix.sh
COPY --from=micromamba /root/.bashrc /root/.bashrc

RUN /usr/local/bin/_dockerfile_setup_root_prefix.sh

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
SHELL ["/usr/local/bin/_dockerfile_shell.sh"]
CMD ["/bin/bash"]

USER root
RUN apt update && apt install -y git curl ffmpeg libsm6 libxext6
RUN micromamba shell init --shell bash && echo "micromamba activate base" >> /root/.bashrc # make root 
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER docker/env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
  micromamba clean --all --yes


