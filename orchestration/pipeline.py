"""Dagster pipeline wiring the dataset index, training and evaluation
stages together. Run locally with:

    dagster dev -f orchestration/pipeline.py

Each asset is a checkpoint: Dagster tracks whether the Rust-built index,
the trained model, and the evaluation report are up to date, and only
reruns what's stale.
"""

import subprocess

import mlflow
from dagster import AssetExecutionContext, Definitions, MetadataValue, asset

IMAGES_DIR = "data/images"
ANNOTATIONS_PATH = "data/annotations.json"
CLASS_NAME = "person"
MODEL_NAME = "perception-detector"


@asset
def dataset_index(context: AssetExecutionContext) -> dict:
    """Builds the Rust dataset index and surfaces per-class counts so
    reviewers can see dataset composition before a training run kicks off."""
    from rust_index import DatasetIndex

    index = DatasetIndex(ANNOTATIONS_PATH)
    counts = index.class_counts()
    context.add_output_metadata({"class_counts": MetadataValue.json(counts)})
    return counts


@asset(deps=[dataset_index])
def trained_model(context: AssetExecutionContext) -> str:
    """Launches distributed training via torchrun and returns the MLflow
    run id so downstream assets can look up the exact model version."""
    result = subprocess.run(
        [
            "torchrun",
            "--nproc_per_node=1",
            "training/train.py",
            "--images-dir",
            IMAGES_DIR,
            "--annotations",
            ANNOTATIONS_PATH,
            "--class-name",
            CLASS_NAME,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    context.log.info(result.stdout)

    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    context.add_output_metadata({"model_version": latest.version, "run_id": latest.run_id})
    return latest.run_id


@asset(deps=[trained_model])
def evaluation_report(context: AssetExecutionContext) -> dict:
    """Pulls eval metrics logged during training and gates promotion:
    only models above the mAP threshold get moved to the 'Staging' stage
    in the MLflow registry, which the serving layer reads from."""
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    run = client.get_run(latest.run_id)
    metrics = run.data.metrics

    passed = metrics.get("mAP", 0.0) >= 0.5
    if passed:
        client.transition_model_version_stage(
            name=MODEL_NAME, version=latest.version, stage="Staging"
        )

    context.add_output_metadata(
        {"metrics": MetadataValue.json(metrics), "promoted": passed}
    )
    return {"metrics": metrics, "promoted": passed}


defs = Definitions(assets=[dataset_index, trained_model, evaluation_report])
