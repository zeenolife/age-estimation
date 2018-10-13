FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ARG PYTHON_VERSION=3.6
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing && \
     /opt/conda/bin/conda clean -ya
ENV PATH /opt/conda/bin:$PATH

RUN conda install -c menpo dlib
RUN conda install pytorch-nightly -c pytorch
RUN conda install -c fastai torchvision-nightly
RUN conda install -c fastai fastai
RUN conda install -c conda-forge opencv

RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx

RUN git clone https://github.com/zeenolife/age-estimation /age-estimation

ENTRYPOINT ["/age-estimation/exec.sh"]
