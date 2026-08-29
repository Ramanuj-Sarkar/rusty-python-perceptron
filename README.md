# ml-infra-project

A miniature version of an ML infrastructure stack for perception model training,
inspired by data-indexing → training → orchestration → deployment pipelines used
in production ML platforms.

## Stack

| Layer         | Tool                         | Where                       |
|---------------|------------------------------|-----------------------------|
| Dataset index | Rust + PyO3 + Maturin        | `rust-index/`               |
| Training      | PyTorch (DDP)                | `training/`                 |
| Orchestration | Dagster                      | `orchestration/`            |
| Observability | MLflow (tracking + registry) | used throughout `training/` |
| Serving       | FastAPI + Docker             | `serving/`                  |
| Deployment    | Kubernetes (Helm chart)      | `deploy/helm/`              |

## Dataset: VisDrone2019-DET

This project targets [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)
(also mirrored on [Hugging Face](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET)):
10,209 drone-captured images across 14 cities, annotated with 10 classes
(pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus,
motor). Its dense scenes and small objects make the Rust index's
class/size-based filtering genuinely useful.

After downloading `VisDrone2019-DET-train` (or `-val`), convert its raw
per-image `.txt` annotations into the single COCO-style JSON this project's
`rust_index.DatasetIndex` expects:

```bash
python scripts/convert_visdrone_to_coco.py \
    --images-dir data/VisDrone2019-DET-train/images \
    --annotations-dir data/VisDrone2019-DET-train/annotations \
    --output data/annotations.json
```

Point `training/train.py` and `orchestration/pipeline.py` at
`data/VisDrone2019-DET-train/images` and `data/annotations.json`.

## Local setup

```bash
# 1. Build the Rust indexing engine and install it into your venv
python -m venv .venv && source .venv/bin/activate
pip install maturin
cd rust-index && maturin develop --release && cd ..

# 2. Install the rest of the stack
pip install -r requirements.txt

# 3. Point MLflow at a local tracking server (or use the default file store)
mlflow server --host 0.0.0.0 --port 5000 &
export MLFLOW_TRACKING_URI=http://localhost:5000

# 4. Launch the Dagster UI to run/inspect the pipeline
dagster dev -f orchestration/pipeline.py
```

Trigger a run from the Dagster UI (`materialize all assets`), or run
`training/train.py` directly for a quick local smoke test.

## Serving

```bash
cd serving
docker build -t perception-inference:local .
docker run -p 8000:8000 -e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI perception-inference:local
```

## Deploying to Kubernetes

```bash
helm install perception-inference deploy/helm \
  --set image.repository=<your-registry>/perception-inference \
  --set image.tag=local
```

## Repo layout

```
rust-index/       Rust crate exposing a fast dataset query engine to Python via PyO3
training/         PyTorch dataset/model/train loop, logs to MLflow
orchestration/    Dagster assets wiring index -> preprocess -> train -> evaluate
serving/          FastAPI inference service + Dockerfile
deploy/helm/      Helm chart for the inference service
```
