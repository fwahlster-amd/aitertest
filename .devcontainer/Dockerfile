#FROM rocm/dev-ubuntu-24.04 this is missing rocThrust/thrust library
FROM rocm/pytorch:latest

# Create non-root user for security
# RUN useradd -m -u 2000 aiter
WORKDIR /workspace
# RUN chown aiter:root /workspace

# Change the default shell to bash
SHELL ["/bin/bash", "-c"]

# Install basic development tools
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    python3-venv

#RUN python3 -m venv /opt/aiterenv
#ENV PATH="/opt/aiterenv/bin:$PATH"

RUN git config --global --add safe.directory /workspace/aiter
RUN git config --global --add safe.directory /workspace/aiter/3rdparty/composable_kernel
#RUN git clone --recursive https://github.com/ROCm/aiter.git /workspace/aiter

#if running on gfx950 node
#ENV GPU_ARCHS=gfx950

# missing from requirements.txt
RUN pip install packaging
RUN pip install pyyaml
RUN pip install setuptools
RUN pip install wheel
#RUN pip install psutil

#RUN pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
#RUN pip install torch --index-url https://download.pytorch.org/whl/nightly/rocm6.4

# Switch to non-root user
# USER aiter