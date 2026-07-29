# vision-infer

ViT-Small fine-tuned on Food-101, served via FastAPI with request batching,
API key auth, and a Docker image built on every push to main.

## Model

| Property | Value |
|---|---|
| Architecture | ViT-Small-patch16-224 (`WinKawaks/vit-small-patch16-224`) |
| Dataset | Food-101 (75 750 train / 25 250 val) |
| Val accuracy | **87.4%** (top-1) after 10 epochs |
| Training time | ~4 h on a single A100 (fp16) |
| W&B run | [vision-infer / food101-vit-small](https://wandb.ai/quinnhasse/vision-infer/runs/food101-vit-small) |

Training script: [`src/train.py`](src/train.py)

## Serving

The API runs at `https://vision-infer.quinnhasse.dev` (requires an API key).

### POST /predict

```bash
curl -X POST https://vision-infer.quinnhasse.dev/predict \
  -H "Authorization: Bearer $API_KEY" \
  -F "image=@photo.jpg"
```

Response:

```json
{
  "predictions": [
    {"label": "pizza", "score": 0.912},
    {"label": "bruschetta", "score": 0.041},
    {"label": "lasagna", "score": 0.023},
    {"label": "spaghetti_bolognese", "score": 0.011},
    {"label": "ravioli", "score": 0.007}
  ],
  "latency_ms": 38.4
}
```

### POST /predict/batch

```bash
curl -X POST https://vision-infer.quinnhasse.dev/predict/batch \
  -H "Authorization: Bearer $API_KEY" \
  -F "images=@a.jpg" \
  -F "images=@b.jpg"
```

### GET /healthz

No auth required. Returns `{"status": "ok"}` when the model is loaded.

## Load test results (50 concurrent users, 60 s, CPU inference)

| Endpoint | p50 | p95 | p99 | RPS |
|---|---|---|---|---|
| `/predict` (single) | 42 ms | 89 ms | 134 ms | 210 |
| `/predict/batch` (4-image) | 61 ms | 118 ms | 175 ms | 52 |

Run the load test yourself:

```bash
pip install locust pillow
locust -f locust/locustfile.py \
  --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 60s --headless
```

## Run locally

```bash
# 1. Install
pip install -r requirements.txt

# 2. Start the server
API_KEY=changeme uvicorn src.server:app --reload

# 3. Send a request
curl -X POST http://localhost:8000/predict \
  -H "Authorization: Bearer changeme" \
  -F "image=@photo.jpg"
```

## Train from scratch

```bash
pip install -r requirements-train.txt
export WANDB_API_KEY=...

python -m src.train \
  --model_name WinKawaks/vit-small-patch16-224 \
  --output_dir ./checkpoints/food101-vit-small \
  --num_train_epochs 10 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5
```

## ONNX export

```bash
python -m src.export_onnx \
  --checkpoint ./checkpoints/food101-vit-small \
  --output ./checkpoints/food101-vit-small.onnx
```

## Docker

```bash
# Build (skips hub download at build time)
docker build --build-arg SKIP_WARMUP=1 -t vision-infer .

# Run
docker run -p 8000:8000 -e API_KEY=changeme vision-infer
```

CI builds and pushes `ghcr.io/quinnhasse/vision-infer-fine-tuned-vit-served-behind-fastapi-docker:latest`
on every merge to main via [`.github/workflows/docker.yml`](.github/workflows/docker.yml).

## Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

12 tests, no external model download required (model is mocked in tests).

## Project layout

```
src/
  config.py          settings from env vars
  model.py           ViT wrapper — load, preprocess, predict
  server.py          FastAPI app — /predict, /predict/batch, /healthz
  train.py           fine-tuning with HuggingFace Trainer + W&B
  export_onnx.py     ONNX export
locust/
  locustfile.py      Locust load test
.github/workflows/
  docker.yml         CI: test → build → push to GHCR
Dockerfile
```
