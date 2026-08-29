"""Convert VisDrone2019-DET's raw per-image annotation .txt files into the
single COCO-style JSON that rust_index.DatasetIndex expects.

Expected input layout (as downloaded from the official VisDrone-Dataset repo):

    VisDrone2019-DET-train/
        images/
            0000001_00000_d_0000001.jpg
            ...
        annotations/
            0000001_00000_d_0000001.txt
            ...

Each annotation line is:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<category_id>,<truncation>,<occlusion>

category_id 0 ("ignored regions") and 11 ("others") are dropped.

Usage:
    python scripts/convert_visdrone_to_coco.py \
        --images-dir data/VisDrone2019-DET-train/images \
        --annotations-dir data/VisDrone2019-DET-train/annotations \
        --output data/annotations.json
"""

import argparse
import json
import struct
from pathlib import Path

# JPEG Start-of-Frame marker types (baseline, progressive, etc). Everything
# else (APPn/EXIF, comments, quant tables, huffman tables, scan data...) is
# skipped without reading its payload.
_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Reads only the marker segments of a JPEG file to find its width and
    height, stopping at the first SOF marker instead of decoding the image.
    Typically reads a few hundred bytes to a few KB, regardless of file size."""
    with open(path, "rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"{path}: not a JPEG (missing SOI marker)")

        while True:
            marker = f.read(2)
            if len(marker) < 2:
                raise ValueError(f"{path}: reached EOF before finding an SOF marker")
            if marker[0] != 0xFF:
                raise ValueError(f"{path}: malformed marker segment")

            marker_type = marker[1]

            # Markers with no payload to skip over.
            if marker_type == 0xD8 or 0xD0 <= marker_type <= 0xD7:
                continue
            if marker_type == 0xD9:
                raise ValueError(f"{path}: reached EOI before finding an SOF marker")

            (segment_len,) = struct.unpack(">H", f.read(2))

            if marker_type in _SOF_MARKERS:
                f.read(1)  # sample precision, unused
                height, width = struct.unpack(">HH", f.read(4))
                return width, height

            # Not the marker we want -- skip its payload (segment_len
            # includes the 2 length bytes we already read).
            f.seek(segment_len - 2, 1)


# Official VisDrone category id -> name mapping. IDs 0 and 11 are excluded below.
VISDRONE_CATEGORIES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


def convert(images_dir: Path, annotations_dir: Path, output_path: Path) -> None:
    images = []
    annotations = []
    categories = [{"id": cid, "name": name} for cid, name in VISDRONE_CATEGORIES.items()]

    ann_files = sorted(annotations_dir.glob("*.txt"))
    if not ann_files:
        raise SystemExit(f"no .txt annotation files found in {annotations_dir}")

    next_ann_id = 1
    skipped_images = 0

    for image_id, ann_file in enumerate(ann_files, start=1):
        image_path = images_dir / f"{ann_file.stem}.jpg"
        if not image_path.exists():
            skipped_images += 1
            continue

        width, height = jpeg_dimensions(image_path)

        images.append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for line in ann_file.read_text().strip().splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(",")
            bbox_left, bbox_top, bbox_w, bbox_h = (float(p) for p in parts[0:4])
            category_id = int(parts[5])

            if category_id not in VISDRONE_CATEGORIES:
                continue  # drop "ignored regions" (0) and "others" (11)

            annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [bbox_left, bbox_top, bbox_w, bbox_h],
                }
            )
            next_ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco))

    print(f"wrote {len(images)} images and {len(annotations)} annotations to {output_path}")
    if skipped_images:
        print(f"skipped {skipped_images} annotation files with no matching image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--annotations-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.images_dir, args.annotations_dir, args.output)
