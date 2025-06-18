# Use NVIDIA's CUDA-enabled PyTorch base image
FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip install --upgrade pip

# Install Dlib (precompiled wheels preferred)
RUN pip install dlib

# Install required packages
RUN pip install \
    numpy \
    opencv-python \
    Pillow \
    torch \
    torchvision \
    torchaudio \
    insightface

# Copy project files into the container (optional)
# COPY . /workspace
COPY ../pre_and_post/blur_faces.py /workspace

ENTRYPOINT ["python", "blur_faces.py"]

# # Default command (can be overridden)
# CMD ["bash"]