# Use official lightweight Python image
# Python 3.11 required by the Keras 3.13.x runtime that saved the model
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files into the container
COPY . .

# Expose the default port required by Hugging Face Spaces
EXPOSE 7860

# Run Uvicorn pointing to your FastAPI instance on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]