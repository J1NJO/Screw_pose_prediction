FROM ubuntu:20.04 as screw_detection
ARG ROS_DISTRO=noetic
ENV DEBIAN_FRONTEND=noninteractive 

# Install basic packages
RUN apt-get update \
&&  apt-get upgrade -y \
&&  apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    wget \
    locales \
    nano \
    git \
    sudo \
    build-essential \
    cmake \
    python3-vtk7 \
    gdb \
    xterm \
    libeigen3-dev \
    libopencv-dev \
    libgl1-mesa-glx \
    python3-pip \
    tzdata \
    qt5-default \
    '^libxcb.*-dev' \
    libx11-xcb-dev \
    libglu1-mesa-dev \
    libxrender-dev \
    libxi-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    xauth \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends apt-utils

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install --upgrade pip && pip install -r requirements.txt

ENV QT_DEBUG_PLUGINS=1

ENV QT_QPA_PLATFORM=xcb

ENV DISPLAY=:0

