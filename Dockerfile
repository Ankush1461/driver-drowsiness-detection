FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install ai-edge-litert first (it pulls in opencv-python)
RUN pip install --no-cache-dir ai-edge-litert

# Force uninstall opencv-python, install contrib version with dnn support
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python-headless && \
    pip install --no-cache-dir opencv-contrib-python-headless

# Install remaining dependencies (skip opencv, already installed)
RUN pip install --no-cache-dir numpy fastapi "uvicorn[standard]" websockets scikit-learn

# Copy application files
COPY . .

# Expose HF spaces default port
EXPOSE 7860

# Run Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
