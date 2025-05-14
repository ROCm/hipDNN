FROM ghcr.io/rocm/therock_build_manylinux_x86_64:main as hipdnn


RUN echo -e "[ROCm-6.4]\n\
name=ROCm6.4\n\
baseurl=https://repo.radeon.com/rocm/rhel8/6.4/main\n\
enabled=1\n\
priority=50\n\
gpgcheck=1\n\
gpgkey=https://repo.radeon.com/rocm/rocm.gpg.key" > /etc/yum.repos.d/rocm.repo

RUN yum install -y rocm-llvm-devel && \
    yum install -y hip-devel

ENV PATH="/opt/rocm:${PATH}"
ENV CMAKE_GENERATOR="Ninja"
