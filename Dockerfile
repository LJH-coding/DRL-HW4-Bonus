FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ARG TORCH_REQUIREMENTS=requirements.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    gnupg \
    git \
    cmake \
    make \
    g++ \
    pkg-config \
    libx11-dev \
    libxxf86vm-dev \
    libxrandr-dev \
    libxi-dev \
    libxext-dev \
    libxrender-dev \
    libxfixes-dev \
    libxss-dev \
    libxinerama-dev \
    libegl1-mesa-dev \
    libgl1-mesa-dev \
    libpulse-dev \
    libudev-dev \
    libevdev-dev \
    libusb-1.0-0-dev \
    libcurl4-openssl-dev \
    libgtk-3-dev \
    libasound2-dev \
    libbluetooth-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswresample-dev \
    libswscale-dev \
    libqt6svg6-dev \
    libvulkan-dev \
    qt6-base-dev \
    qt6-base-dev-tools \
    qt6-base-private-dev \
    qt6-tools-dev \
    qt6-tools-dev-tools \
    python3-apt \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository universe \
    && add-apt-repository ppa:deadsnakes/ppa \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && curl -fsSL https://apt.kitware.com/keys/kitware-archive-latest.asc \
      | gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main" \
      > /etc/apt/sources.list.d/kitware.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      gcc-13 \
      g++-13 \
      cmake \
      python3.12 \
      python3.12-dev \
      python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /workspace
COPY . /workspace

RUN python -m pip install --no-cache-dir -r ${TORCH_REQUIREMENTS}
RUN bash scripts/build-dolphin-linux.sh
RUN chmod +x scripts/run_headless_linux.sh

ENTRYPOINT ["bash", "scripts/run_headless_linux.sh"]
CMD ["python", "BTR.py"]
