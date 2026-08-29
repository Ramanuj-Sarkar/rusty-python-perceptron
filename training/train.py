"""Distributed training entrypoint. Launch with:

    torchrun --nproc_per_node=2 training/train.py \
        --images-dir data/images --annotations data/annotations.json

Logs params/metrics/model artifacts to MLflow and registers the trained
model in the MLflow Model Registry, which is what the Dagster `train`
and `evaluate` assets call under the hood.
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from dataset import PerceptionDataset
from model import build_model


def setup_ddp():
    """No-op locally (single process); real DDP init when launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None
        return local_rank
    return 0


def train(args):
    local_rank = setup_ddp()
    is_distributed = dist.is_initialized()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dataset = PerceptionDataset(
        images_dir=args.images_dir,
        annotations_path=args.annotations,
        class_name=args.class_name,
        transform=transforms.ToTensor(),
    )
    sampler = DistributedSampler(dataset) if is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    model = build_model(num_classes=args.num_classes).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Only the rank-0 process talks to MLflow to avoid duplicate runs.
    is_main = local_rank == 0
    if is_main:
        mlflow.set_experiment("perception-detector")
        mlflow.start_run()
        mlflow.log_params(vars(args))

    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        for images, _file_names in loader:
            images = [img.to(device) for img in images]
            # Real project: build targets (boxes/labels) from the Rust index
            # output and pass them to the model alongside `images`.
            optimizer.zero_grad()
            # loss_dict = model(images, targets); loss = sum(loss_dict.values())
            # loss.backward(); optimizer.step(); running_loss += loss.item()

        if is_main:
            mlflow.log_metric("train_loss", running_loss / max(len(loader), 1), step=epoch)

    if is_main:
        # Registers the model under "perception-detector" so the evaluate
        # asset and the serving layer can pull the latest version by name.
        mlflow.pytorch.log_model(
            model.module if is_distributed else model,
            artifact_path="model",
            registered_model_name="perception-detector",
        )
        mlflow.end_run()

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--class-name", default="person")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    train(parser.parse_args())
