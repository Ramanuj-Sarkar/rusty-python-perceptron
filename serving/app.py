"""FastAPI inference service. Loads the current 'Staging' model version
from the MLflow registry on startup so a new promoted model can be picked
up by restarting the deployment -- no code change or rebuild needed."""

import io

import mlflow.pytorch
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

MODEL_NAME = "perception-detector"
MODEL_STAGE = "Staging"

app = FastAPI(title="perception-inference")
model = None


@app.on_event("startup")
def load_model():
    global model
    model = mlflow.pytorch.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    model.eval()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = transforms.ToTensor()(image)

    with torch.no_grad():
        [prediction] = model([tensor])

    return {
        "boxes": prediction["boxes"].tolist(),
        "labels": prediction["labels"].tolist(),
        "scores": prediction["scores"].tolist(),
    }
