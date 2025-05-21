# rocm/dev-ubuntu-24.04 this is missing rocThrust/thrust library
FROM rocm/pytorch:latest

WORKDIR /workspace

# Change the default shell to bash
SHELL ["/bin/bash", "-c"]

# Install basic development tools
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    python3-venv


RUN git config --global --add safe.directory /workspace/aiter
RUN git config --global --add safe.directory /workspace/aiter/3rdparty/composable_kernel

# missing from requirements.txt
RUN pip install packaging
RUN pip install pyyaml
RUN pip install setuptools
RUN pip install wheel