# Verwende aktuelles Runpod Base Image mit PyTorch vorinstalliert
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Arbeitsverzeichnis
WORKDIR /app

# System Dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python Dependencies installieren (ohne torch - ist schon im Base Image!)
# Files sind jetzt im serverless/ Unterverzeichnis
COPY serverless/requirements.txt .
RUN pip install --no-cache-dir \
    diffusers>=0.30.0 \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    runpod>=1.6.0 \
    pillow>=10.0.0 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0

# LongCat-Video Repository clonen und installieren
# Cache-Break: Verwende --no-cache beim Build wenn du neu clonen willst
RUN rm -rf /app/LongCat-Video && \
    git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git /app/LongCat-Video && \
    cd /app/LongCat-Video && \
    pip install --no-cache-dir --ignore-installed blinker -r requirements.txt

# Handler kopieren (NACH git clone, damit Code-Änderungen den Cache brechen!)
COPY serverless/handler.py .

# Environment für HuggingFace Cache
# Runpod Network Storage wird unter /runpod-volume gemountet
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/transformers
ENV DIFFUSERS_CACHE=/runpod-volume/huggingface/diffusers

# Model weights OPTIONAL vorab herunterladen (spart Zeit beim ersten Start)
# WARNUNG: Das macht das Image ~30GB größer!
# Auskommentieren wenn du das Model zur Build-Zeit laden willst:
# RUN pip install --no-cache-dir huggingface_hub && \
#     huggingface-cli download meituan-longcat/LongCat-Video --cache-dir /app/hf_cache

# Runpod erwartet handler.py im Root
ENV PYTHONUNBUFFERED=1

# Test: Model laden beim Build (optional)
# RUN python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Entry Point
CMD ["python", "-u", "handler.py"]
