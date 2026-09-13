FROM python:3.11-slim

# Install system dependencies including FFmpeg.
# (libgl1/libglib2.0-0/libsm6/libxext6/libxrender-dev were only ever
# needed by opencv -- dropped along with torch/ultralytics/opencv in
# favor of onnxruntime, see requirements.txt.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Most managed container platforms (Render, HF Spaces, Cloud Run)
# run the container as a non-root user and only guarantee write
# access under that user's home directory -- so the app and its
# writable data dirs need to live under /home/user instead of /app
# as root.
RUN useradd -m -u 1000 user

# WORKDIR creates its directory as root even with USER already set
# below it -- chown it explicitly while still root, or the non-root
# user can create files here (via COPY, which runs with elevated
# privileges regardless of USER) but can't create brand-new
# subdirectories of its own (e.g. the mkdir -p further down).
WORKDIR /home/user/app
RUN chown -R user:user /home/user/app

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

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
