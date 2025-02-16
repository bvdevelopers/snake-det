# Use a Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install system dependencies for TensorFlow Lite (e.g., libsndfile, PIL dependencies)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install FastAPI, Uvicorn, TensorFlow, and other dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and model into the container
COPY . /app

# Expose port 8000 for FastAPI
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
