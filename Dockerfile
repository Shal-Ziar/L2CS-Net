# Use NVIDIA CUDA base image with cuDNN for GPU support
ARG BUILDPLATFORM=linux/amd64
FROM --platform=${BUILDPLATFORM} docker.io/python:3.13-slim-bullseye AS base
ARG GROUP
RUN echo "Building for platform: ${BUILDPLATFORM} with group: ${GROUP}"
WORKDIR /app
RUN mkdir -p /.cache
ENV UV_PROJECT_ENVIRONMENT=/usr/local/
# RUN apt-get update && apt-get install -y \
#     python3-uv
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install uv
# Set Python 3.10 as default
COPY pyproject.toml pyproject.toml
COPY uv.lock uv.lock
COPY l2cs/ ./l2cs/
COPY demo.py train.py test.py leave_one_out_eval.py vector_output.py ./
RUN --mount=type=cache,target=/root/.cache uv sync --no-dev
RUN --mount=type=cache,target=/root/.cache uv sync --group ${GROUP}


# Copy additional Python scripts (optional - uncomment if needed)

# Install Python dependencies


# Create directories for models and datasets
RUN mkdir -p /app/models /app/datasets

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - can be overridden
CMD ["/bin/bash"]
