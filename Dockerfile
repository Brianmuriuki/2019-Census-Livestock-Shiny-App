# syntax=docker/dockerfile:1
FROM python:3.8-slim-buster


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libssl-dev \
        libffi-dev \
        libpq-dev \
        git \
        && rm -rf /var/lib/apt/lists/*

# Install dependencies
# Install dependencies
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Copy application code
COPY analysis.py /

# Set working directory
WORKDIR /

# Run the application
CMD ["python", "/analysis.py"]