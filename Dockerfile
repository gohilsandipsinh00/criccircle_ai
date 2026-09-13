FROM python:3.11-slim

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Most managed container platforms (Render, HF Spaces, Cloud Run)
# run the container as a non-root user and only guarantee write
# access under that user's home directory -- so the app, its writable
# data dirs, and every library that caches config/state (ultralytics,
# matplotlib) all need to live under /home/user instead of /app as
# root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics \
    MPLCONFIGDIR=/home/user/.config/matplotlib

WORKDIR /home/user/app

# Install Python dependencies (--user since we're not root)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy app code
COPY --chown=user . .

# Create directories
RUN mkdir -p models/yolo models/classifier \
    logs assets data/raw data/frames \
    data/annotated data/processed

EXPOSE 8001

# Single worker: each uvicorn worker loads its own copy of the YOLO
# and classifier models into memory, and free-tier hosts don't have
# RAM to spare for a second copy.
#
# Render (and most PaaS Docker hosts) assign the listen port at
# runtime via $PORT rather than a fixed one -- shell form so that
# expands, falling back to 8001 for local `docker run` / HF Spaces
# where $PORT isn't set.
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"
