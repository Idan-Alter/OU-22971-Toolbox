FROM continuumio/miniconda3:26.1.1-1

WORKDIR /workspace

# Compose builds from the Ray part root, so this copies the part-local env spec.
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy
