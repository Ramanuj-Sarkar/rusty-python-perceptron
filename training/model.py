"""Model definition: a torchvision detection backbone fine-tuned for the
target class set. Swappable for a custom architecture without touching
the training loop or orchestration layer."""

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_model(num_classes: int):
    """Returns a Faster R-CNN model with a fresh classification head sized
    for `num_classes` (including background)."""
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
