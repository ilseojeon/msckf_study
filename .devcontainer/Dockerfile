# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.205.2/containers/cpp/.devcontainer/base.Dockerfile

# [Choice] Debian / Ubuntu version (use Debian 11/9, Ubuntu 18.04/21.04 on local arm64/Apple Silicon): debian-11, debian-10, debian-9, ubuntu-21.04, ubuntu-20.04, ubuntu-18.04
ARG VARIANT="bullseye"
FROM mcr.microsoft.com/vscode/devcontainers/cpp:0-${VARIANT}

RUN apt-get -y update && apt-get install -y --no-install-recommends libeigen3-dev

RUN cd /tmp && git clone https://gitlab.com/libeigen/eigen.git eigen && \
cd eigen && git checkout 3.4.0 && \
mkdir build && cd build && \
cmake .. && \
make -j2 && make install && \
ln -s /usr/include/eigen3/Eigen /usr/include/Eigen && \
ln -s /usr/local/include/eigen3/Eigen /usr/local/include/Eigen && \
rm -rf /tmp/*

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
