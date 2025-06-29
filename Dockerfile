# Dockerfile for NYC Itinerary Planning System
# Based on research reproducibility requirements

FROM python:3.8.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    gdal-bin \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p data benchmarks/results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose Streamlit port
EXPOSE 8501

# Default command runs benchmarks
CMD ["python", "benchmarks/benchmark_runner.py"]

# Alternative commands:
# For Streamlit demo: 
# CMD ["streamlit", "run", "streamlit_demo.py", "--server.port=8501", "--server.address=0.0.0.0"]
# For data preparation:
# CMD ["python", "prepare_nyc_data.py"]