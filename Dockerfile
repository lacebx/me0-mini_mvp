# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libopenblas-dev \
        git \
        && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app/

# Expose the port (Cloud Run uses $PORT)
ENV PORT=8080

# Command to run the app with Gunicorn
CMD exec gunicorn chatbotv1:app --bind 0.0.0.0:$PORT