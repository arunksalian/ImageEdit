# Use Python 3.9 as base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-osd \
    poppler-utils \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ghostscript \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev \
    libicu-dev \
    libpango1.0-dev \
    libcairo2-dev \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Find and copy Tesseract data files
RUN find /usr -name "eng.traineddata" -type f -exec ls -l {} \; && \
    TESSDATA_SRC=$(find /usr -name "eng.traineddata" -type f -exec dirname {} \;) && \
    echo "Found Tesseract data at: $TESSDATA_SRC" && \
    mkdir -p /usr/share/tesseract-ocr/4.00/tessdata && \
    cp $TESSDATA_SRC/eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/ && \
    cp $TESSDATA_SRC/osd.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

# Set environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV PYTHONPATH=/app
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV FLASK_DEBUG=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV TESSERACT_CONFIG="--oem 3 --psm 6 -c preserve_interword_spaces=1 -c textord_heavy_nr=1"
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PYTHONHASHSEED=0
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1
ENV PYTHONOPTIMIZE=2

# Verify Tesseract installation and configuration
RUN tesseract --version && \
    echo "Tesseract data directory: $TESSDATA_PREFIX" && \
    ls -la $TESSDATA_PREFIX/eng.traineddata && \
    tesseract --list-langs

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    pytesseract==0.3.10 \
    pdf2image==1.16.3 \
    PyMuPDF==1.22.5 \
    Pillow==10.0.0 \
    numpy==1.24.3 \
    opencv-python-headless==4.8.0.74 \
    gunicorn==21.2.0 \
    gevent==23.7.0 \
    greenlet==3.0.1

# Create necessary directories with proper permissions
RUN mkdir -p /app/uploads /app/logs /app/tests /app/temp /app/redacted /tmp/tesseract && \
    chmod -R 777 /app/uploads /app/logs /app/tests /app/temp /app/redacted /tmp/tesseract

# Copy the rest of the application
COPY . .

# Set permissions for copied files
RUN chmod -R 777 /app

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "test" ]; then\n\
    echo "Running tests..."\n\
    pytest -v --cov=app --cov-report=term-missing tests/\n\
elif [ "$1" = "test-watch" ]; then\n\
    echo "Running tests in watch mode..."\n\
    pytest-watch -v --cov=app --cov-report=term-missing tests/\n\
else\n\
    echo "Starting application..."\n\
    echo "Current directory: $(pwd)"\n\
    echo "Directory contents:"\n\
    ls -la\n\
    echo "Python path:"\n\
    python -c "import sys; print(sys.path)"\n\
    echo "Starting Gunicorn..."\n\
    gunicorn --bind 0.0.0.0:5000 \
             --workers $(nproc) \
             --threads 4 \
             --worker-class gevent \
             --timeout 120 \
             --log-level info \
             --error-logfile /app/logs/gunicorn-error.log \
             --access-logfile /app/logs/gunicorn-access.log \
             --capture-output \
             app:app\n\
fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 5000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command to run the application
CMD ["run"] 