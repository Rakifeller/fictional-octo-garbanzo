FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive         PYTHONUNBUFFERED=1         HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends git &&         rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps (torch comes from base image)
COPY serverless/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY app ./app
COPY serverless/worker.py ./serverless/worker.py

# Start serverless worker
CMD ["python", "-u", "serverless/worker.py"]
