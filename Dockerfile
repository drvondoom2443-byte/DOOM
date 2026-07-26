FROM python:3.10-slim

# Install system dependencies required by MoviePy / FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    imagemagick \
    fonts-liberation \
    && rm -rf /var/lib/apt-get/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker layer caching correctly
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port 7860
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
