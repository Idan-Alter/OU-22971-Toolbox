FROM continuumio/miniconda3:26.1.1-1

SHELL ["/bin/bash", "-lc"]
WORKDIR /workspace

COPY environment.yml /tmp/22971-grading-environment.yml

RUN conda env create -f /tmp/22971-grading-environment.yml \
    && conda clean -afy \
    && printf '\n# Load Conda in interactive Bash shells.\nif [ -f /opt/conda/etc/profile.d/conda.sh ]; then\n    . /opt/conda/etc/profile.d/conda.sh\nfi\nconda activate 22971-grading\n' >> /etc/bash.bashrc

ENV CONDA_DEFAULT_ENV=22971-grading
ENV PATH=/opt/conda/envs/22971-grading/bin:/opt/conda/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV USER=grader
ENV USERNAME=grader
ENV METAFLOW_USER=grader

CMD ["bash"]
