FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install all other dependencies first (some may pull in a minimal opencv)
RUN pip install --no-cache-dir numpy fastapi "uvicorn[standard]" websockets scikit-learn

# Install ai-edge-litert WITHOUT its opencv dependency
RUN pip install --no-cache-dir --no-deps ai-edge-litert

# CRITICAL: Purge ALL opencv variants to prevent conflicts that break cv2.dnn.readNetFromCaffe
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless 2>/dev/null; true

# Install the SINGLE correct opencv package with full dnn support (Caffe, ONNX, etc.)
RUN pip install --no-cache-dir opencv-contrib-python-headless

# Copy application files
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
