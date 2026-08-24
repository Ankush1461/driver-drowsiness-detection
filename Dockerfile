FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install ai-edge-litert WITHOUT its opencv dependency
RUN pip install --no-cache-dir --no-deps ai-edge-litert

# Install opencv-contrib with dnn support (no conflict now)
RUN pip install --no-cache-dir opencv-contrib-python-headless

# Install all other dependencies
RUN pip install --no-cache-dir numpy fastapi "uvicorn[standard]" websockets scikit-learn

# Copy application files
COPY . .

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
