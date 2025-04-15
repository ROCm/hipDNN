FROM ubuntu:22.04 as hipdnn

ARG DEBIAN_FRONTEND=noninteractive

ENV APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=DontWarn

# Support multiarch
RUN dpkg --add-architecture i386

# Install preliminary dependencies and add rocm gpg key
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
        apt-utils ca-certificates curl libnuma-dev gnupg2 wget  && \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm-keyring.gpg

# Get and install amdgpu-install.
RUN wget https://repo.radeon.com/amdgpu-install/6.3.2/ubuntu/jammy/amdgpu-install_6.3.60302-1_all.deb --no-check-certificate && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
       ./amdgpu-install_6.3.60302-1_all.deb

# Add rocm repository
RUN export ROCM_APT_VER=6.3.4; \
    echo $ROCM_APT_VER &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/amdgpu/$ROCM_APT_VER/ubuntu jammy main > /etc/apt/sources.list.d/amdgpu.list' &&\
    sh -c 'echo deb [arch=amd64 signed-by=/etc/apt/trusted.gpg.d/rocm-keyring.gpg] https://repo.radeon.com/rocm/apt/$ROCM_APT_VER jammy main > /etc/apt/sources.list.d/rocm.list'

RUN sh -c "echo deb http://mirrors.kernel.org/ubuntu jammy main universe | tee -a /etc/apt/sources.list" && \
    amdgpu-install -y --usecase=rocm --no-dkms

# Install dependencies
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --allow-unauthenticated \
    build-essential \
    cmake \
    clang-format-12 \
    doxygen \
    gdb \
    git \
    git-lfs \
    lbzip2 \
    lcov \
    libncurses5-dev \
    stunnel \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-pybind11 \
    redis \
    rocm-developer-tools \
    rocm-llvm-dev \
    rpm \
    software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf amdgpu-install* && \
# Remove unnecessary rocm components
    apt-get remove -y miopen-hip

RUN pip3 install --upgrade cmake==3.27.5

ENV PATH="/opt/rocm:${PATH}"