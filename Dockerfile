# Dockerfile for FractalMIDI Web Application

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/
COPY web/requirements_web.txt /app/web/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r web/requirements_web.txt

# Copy application code
COPY . /app/

# Create directories
RUN mkdir -p /app/outputs /app/logs /app/dataset/validation_examples /app/web/outputs

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Run the application
CMD ["python", "web/backend/app.py"]

