FROM continuumio/miniconda3:26.1.1-1

WORKDIR /workspace

COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy && \
    printf '\n# Load Conda in interactive Bash shells.\nif [ -f /opt/conda/etc/profile.d/conda.sh ]; then\n    . /opt/conda/etc/profile.d/conda.sh\nfi\n' >> /etc/bash.bashrc
