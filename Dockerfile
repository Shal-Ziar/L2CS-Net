# Use NVIDIA CUDA base image with cuDNN for GPU support
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.10 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Copy project files
COPY pyproject.toml LICENSE README.md ./
COPY l2cs/ ./l2cs/

# Copy additional Python scripts (optional - uncomment if needed)
COPY demo.py train.py test.py leave_one_out_eval.py ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Create directories for models and datasets
RUN mkdir -p /app/models /app/datasets

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden
CMD ["/bin/bash"]
