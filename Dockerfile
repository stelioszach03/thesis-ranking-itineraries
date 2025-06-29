# NYC Itinerary Demo Docker Image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY demo/ ./demo/
COPY data/ ./data/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install -e .

# Prepare NYC data
RUN python src/prepare_nyc_data.py

# Expose Flask port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=demo/demo_nyc.py
ENV PYTHONUNBUFFERED=1

# Run the demo
CMD ["python", "demo/demo_nyc.py"]