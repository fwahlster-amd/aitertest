FROM rocm/dev-ubuntu-24.04

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
    python3-venv

RUN python3 -m venv /opt/aiterenv
ENV PATH="/opt/aiterenv/bin:$PATH"
#RUN source /opt/aiterenv/bin/activate
# missing from requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
RUN pip install packaging 
#RUN pip install psutil

COPY requirements.txt .
RUN pip install -r requirements.txt 

#RUN --mount=type=bind,source=/,target=/workspace/ pip install -r requirements.txt

# ModuleNotFoundError: No module named 'aiter.jit.module_aiter_enum'

# Switch to non-root user
# USER aiter