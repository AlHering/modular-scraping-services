FROM ubuntu:22.04
ENV PYTHONUNBUFFERED=1

# Setting up basic repo 
ARG DEBIAN_FRONTEND noninteractive
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Setting up working directory
ADD ./ modular-scraping-services/
WORKDIR /modular-scraping-services
ENV RUNNING_IN_DOCKER=True
ENV CUDA_SUPPORT=False

# Install prerequisites
RUN apt-get update && apt-get install -y apt-utils \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get update && apt-get install -y \
    make build-essential wget curl git nano ffmpeg libsm6 libxext6 \
    p7zip-full p7zip-rar \
    python3.10 python3.10-distutils python3.10-dev python3.10-venv \
    libgoogle-perftools4 libtcmalloc-minimal4 libgoogle-perftools-dev \
    pkg-config libcairo2-dev libjpeg-dev libgif-dev && apt-get clean -y

# Project prerequisites
RUN apt-get update && apt-get install -y \
    portaudio19-dev libportaudio2 python3-pyaudio \
    libssl-dev libasound2 libsndfile1 espeak && apt-get clean -y

# Create venv
RUN if [ ! -d "venv" ]; \
    then \
    python3.10 -m venv venv; \
    fi

# Setup
RUN /bin/bash install.sh

# Command for starting modular-scraping-services
CMD ["/bin/bash", "run.sh"]