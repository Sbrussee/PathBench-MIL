# Use an NVIDIA CUDA base image for GPU support.
# When running the container, use the NVIDIA runtime (e.g., --gpus all).
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Prevent interactive dialogs during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, Python 3.9, and Git.
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev && \
    # Install pip for Python 3.9
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app


# --------------------------------------------------
# Build steps for PathBench-MIL
# --------------------------------------------------
COPY . /app/PathBench-MIL
WORKDIR /app/PathBench-MIL

# Run setup_pathbench.py to create the virtual environment and install base tools.
RUN python3.9 setup_pathbench.py

# "Activate" the virtual environment for subsequent commands:
ENV VIRTUAL_ENV=/app/PathBench-MIL/pathbench_env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# --------------------------------------------------
# Install the slideflow fork package
# --------------------------------------------------
WORKDIR /app/PathBench-MIL/slideflow_fork
RUN pip install -e .

# --------------------------------------------------
# Install the PathBench-MIL package
# --------------------------------------------------
WORKDIR /app/PathBench-MIL
RUN pip install -e .

# Ensure run_pathbench.sh is executable
RUN chmod +x run_pathbench.sh

# Set backend environment variables (can also be overridden at runtime)
ENV SF_SLIDE_BACKEND=cucim
ENV SF_BACKEND=torch

# Unbuffer Python stdout/stderr so logs appear immediately
ENV PYTHONUNBUFFERED=1

# Set the default command.
# In interactive mode, you can override this to run bash instead.
CMD ["bash", "run_pathbench.sh"]
