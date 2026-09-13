---
title: CricCircle AI Service
emoji: 🏏
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8001
pinned: false
---

# CricCircle AI Service

AI-powered cricket highlight detection system.
Detects SIX, FOUR, WICKET, CATCH, CELEBRATION moments
from uploaded cricket match videos.

## Tech Stack
- FastAPI (Python web framework)
- YOLOv8 (object detection)
- ResNet50 (event classification)
- FFmpeg (video processing)
- librosa (audio analysis)
- AWS S3 (video storage)
- Firebase FCM (push notifications)
- Redis (job queue)
- PostgreSQL (results database)

## Quick Start

### 1. Install FFmpeg
- Windows: `choco install ffmpeg`
- Mac: `brew install ffmpeg`
- Linux: `apt install ffmpeg`

### 2. Setup Python environment
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```
Fill in your AWS, Firebase, and database credentials.

### 4. Run service
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 5. View API docs
http://localhost:8001/docs

## Model Training

### Train YOLO (object detection)
```bash
python app/ml/training/train_yolo.py --data dataset/cricket.yaml --epochs 100
```

### Train Classifier (event classification)
```bash
python app/ml/training/train_classifier.py --data dataset/classifier --epochs 50
```

## API Endpoints

- `POST /api/v1/videos/process` — Start AI processing of uploaded video
- `GET /api/v1/videos/status/{video_id}` — Get processing progress
- `GET /api/v1/videos/{video_id}/highlights` — Get detected highlights
- `POST /api/v1/videos/highlights/update` — User confirms/rejects AI suggestion
- `POST /api/v1/videos/{video_id}/retry-ai` — Retry failed processing
- `POST /api/v1/webhooks/nestjs/upload-complete` — NestJS-triggered processing webhook
- `GET /health` — Service health check

## Processing Pipeline

1. Flutter app uploads video to S3
2. NestJS backend calls `POST /api/v1/videos/process`
3. AI service downloads video from S3
4. FFmpeg extracts audio + frames
5. Audio analyzer finds excitement spikes
6. YOLOv8 detects objects in frames
7. Combined detections find highlight moments
8. ResNet50 classifies each moment type
9. FFmpeg cuts 6-second highlight clips
10. CricCircle watermark added to clips
11. Clips uploaded to S3
12. FCM notification sent to user
13. NestJS webhook called with results
14. Flutter app shows AI Review screen

## Deployment (free demo hosting: Hugging Face Spaces)

This repo is set up to deploy as-is to a Hugging Face Space using the
Docker SDK (see the front matter at the top of this file).

1. Create a Space at huggingface.co/new-space with **SDK: Docker**
   and hardware **CPU basic** (free).
2. In the Space's **Settings -> Repository secrets**, set:
   `DATABASE_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `NESTJS_WEBHOOK_SECRET` (placeholder values are fine -- none of
   these are actually used while `USE_S3=false`), plus
   `USE_S3=false`, `USE_LOCAL_STORE=true`, and `PUBLIC_BASE_URL` set
   to the Space's own URL (`https://<user>-<space>.hf.space`).
3. Push this repo to the Space's git remote (shown on the Space
   page) to trigger a build.
4. Once built, the API is live at that same public URL --
   `/health` and `/docs` should respond immediately.

## Project Structure

See `app/` for the FastAPI application (routes, core pipeline, ML
wrappers, services, utils), `app/ml/training/` for model training
scripts, `scripts/` for dataset tooling, and `tests/` for the test
suite.
