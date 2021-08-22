# Dockerfile based on https://github.com/timesler/docker-jupyter-dl-gpu
# TODO make sure correct CONDA_VERSION is used.

FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04

LABEL maintainer="us4us ltd. <support@us4us.eu>"
USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    build-essential \
    git \
    vim \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    pandoc \
    python-dev \
    tini \
#    texlive-fonts-extra \
#    texlive-fonts-recommended \
#    texlive-generic-recommended \
    texlive-latex-base \
    texlive-latex-extra \
#    texlive-xetex \
    tzdata \
    unzip \
    nano \
    ffmpeg \
    imagemagick\
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV CONDA_DIR=/opt/conda
ENV SHELL_NAME=bash
ENV SHELL="/bin/${SHELL_NAME}"
ENV IMAGE_USER="student"
ENV USER_UID=1000
ENV USER_GID=100
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH
ENV HOME="/home/${IMAGE_USER}"

ENV PYTHON_VERSION=3.8 \
    MINICONDA_VERSION=py38_4.9.2 \
    MINICONDA_MD5=122c8c9beb51e124ab32a0fa6426c656 \
    CONDA_VERSION=4.9.2 \
    GPU_SHORT_COURSE_VERSION=1.0 \
    GPU_SHORT_COURSE_BRANCH=develop \
    GPU_SHORT_COURSE_REPO_NAME=ius-2021-gpu-short-course

# Add a script that we will use to correct permissions after running certain commands
ADD utils/fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

RUN useradd -m -s /bin/bash -N -u $USER_UID $IMAGE_USER && \
    mkdir -p $CONDA_DIR && \
    chown $IMAGE_USER:$IMAGE_GID $CONDA_DIR && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME && \
    fix-permissions "$(dirname $CONDA_DIR)"

USER $USER_UID
WORKDIR $HOME

# Setup work directory for backward-compatibility
RUN git clone --branch develop https://github.com/us4useu/${GPU_SHORT_COURSE_REPO_NAME}.git && \
    fix-permissions /home/$IMAGE_USER

RUN cd /tmp && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_MD5} *Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    $CONDA_DIR/bin/conda config --system --prepend channels conda-forge && \
    $CONDA_DIR/bin/conda config --system --set auto_update_conda false && \
    $CONDA_DIR/bin/conda config --system --set show_channel_urls true

WORKDIR $HOME

# Install short-course requirements.
RUN $CONDA_DIR/bin/conda env update --name base --file ${GPU_SHORT_COURSE_REPO_NAME}/conda-env-linux.yml --prune && \
    $CONDA_DIR/bin/pip install ./${GPU_SHORT_COURSE_REPO_NAME} && \
    $CONDA_DIR/bin/conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
#    $CONDA_DIR/bin/conda install --quiet --yes conda && \
#    $CONDA_DIR/bin/conda update --all --quiet --yes && \
    $CONDA_DIR/bin/conda clean --all -f -y && \
    $CONDA_DIR/bin/jupyter notebook --generate-config && \
    # jupyter lab extensions
    $CONDA_DIR/bin/jupyter labextension install @aquirdturtle/collapsible_headings && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$IMAGE_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$IMAGE_USER


EXPOSE 8888

# Configure container startup
WORKDIR $HOME/${GPU_SHORT_COURSE_REPO_NAME}
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh", "--no-browser"]

# Add local files as late as possible to avoid cache busting
COPY utils/start.sh /usr/local/bin/
COPY utils/start-notebook.sh /usr/local/bin/
COPY utils/jupyter_notebook_config.py /etc/jupyter/

# Fix permissions on /etc/jupyter as root
USER root
RUN fix-permissions /etc/jupyter/
RUN chmod +x /usr/local/bin/start-notebook.sh
RUN chmod +x /usr/local/bin/start.sh

USER $USER_UID

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME /home/$IMAGE_USER/.cache/
RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" && \
    fix-permissions /home/$IMAGE_USER

USER $USER_UID