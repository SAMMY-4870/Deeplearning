# AI vs Real Media Detector

A full-stack project to detect whether an uploaded image or video is likely AI-generated (deepfake/synthetic) or real.

- Backend: FastAPI (Python) + PyTorch
- Models: Xception (images), Xception + LSTM (videos)
- Frontend: Simple HTML/JS uploader

Note: Models ship untrained by default. You must train on your dataset to get meaningful predictions.

## Setup

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2. (Optional) Set environment variables to point to your trained weights:

```bash
export IMAGE_WEIGHTS_PATH=/workspace/checkpoints/xception_image.pt
export VIDEO_WEIGHTS_PATH=/workspace/checkpoints/video_lstm.pt
```

## Run API

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Open the frontend by serving the `frontend` folder in any static server, or simply open `frontend/index.html` in your browser and set `API_BASE` if needed:

```html
<script>
  window.API_BASE = 'http://localhost:8000';
</script>
```

## Train

Prepare datasets with the following structure:

```
/your_image_dataset
  /real
    img1.jpg, ...
  /ai
    img2.jpg, ...

/your_video_dataset
  /real
    clip1.mp4, ...
  /ai
    clip2.mp4, ...
```

Then run:

```bash
# Image model
python backend/train/train_image.py --data /your_image_dataset --out /workspace/checkpoints/xception_image.pt --epochs 3

# Video model
python backend/train/train_video.py --data /your_video_dataset --out /workspace/checkpoints/video_lstm.pt --epochs 2 --num_frames 32
```

After training, restart the API with the environment variables pointing to the saved weights.

## Endpoints

- POST `/predict/image` - form-data field `file` (JPEG/PNG/WebP)
- POST `/predict/video` - form-data field `file` (MP4/MKV/AVI/MOV), query or form `num_frames` optional

Both return:

```json
{ "label": "AI-generated" | "Real", "is_ai": true|false, "confidence": 0.0-1.0, "detail": "optional warning" }
```

## Notes

- For production, secure CORS and file handling, add rate limiting, logging, and monitoring.
- Consider augmentations and class balancing for better training.
- Add calibration or a threshold sweep to tune precision/recall depending on your risk appetite.