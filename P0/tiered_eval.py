#!/usr/bin/env python3
"""Standalone area-tiered AP evaluation for VisDrone.

Runs batched inference over the val split with a trained checkpoint, then computes
COCO-style small/medium/large AP (plus a finer "tiny" tier) from GT box area. A detection is
bucketed by the area of the GT it matches; detections that match nothing are false positives in
every tier, and detections matched to a GT outside a tier are ignored for that tier (COCO
semantics). Overall mAP and the tiered APs come from this single code path, so the table is
self-consistent (and reused for the P1 on/off comparison).

Tiering (pixel area of GT box; thresholds configurable):
    tiny   < 16^2
    small  16^2 .. 32^2
    medium 32^2 .. 96^2
    large  >= 96^2

Usage:
    python P0/tiered_eval.py --weights runs/p0/visdrone-baseline-yolo11n/weights/best.pt \
                             --imgsz 640 --device 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils import YAML

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "datasets" / "VisDrone.yaml"

IOU_THRS = np.round(np.linspace(0.5, 0.95, 10), 2)
# (label, area_low, area_high); area_high=None means open-ended upper bound.
DEFAULT_TIERS = [
    ("tiny", 0, 16**2),
    ("small", 16**2, 32**2),
    ("medium", 32**2, 96**2),
    ("large", 96**2, None),
]


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (Na,4) and (Nb,4) xyxy boxes -> (Na, Nb)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a, b = a[:, None, :], b[None, :, :]
    inter = (
        np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
        * np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    )
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)
    return np.where(area_a + area_b - inter > 0, inter / (area_a + area_b - inter), 0.0)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    idx = np.linspace(0.0, 1.0, 101)
    prec = np.concatenate([[0.0], prec, [0.0]])
    rec = np.concatenate([[0.0], rec, [1.0]])
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    return float(np.trapezoid(np.interp(idx, rec, prec), idx))


def _img_sizes(images_dir: Path) -> dict[str, tuple[int, int]]:
    """Map image stem -> (W, H) pixel size (needed to denormalize GT boxes)."""
    sizes = {}
    for p in images_dir.glob("*.jpg"):
        sizes[p.stem] = Image.open(p).size  # (W, H)
    return sizes


def _load_gt(labels_dir: Path, sizes: dict[str, tuple[int, int]]) -> dict[str, list[tuple[int, np.ndarray, float]]]:
    """Return {stem: [(cls, xyxy, area_px)]} from YOLO-format labels."""
    gts: dict[str, list[tuple[int, np.ndarray, float]]] = {}
    for p in labels_dir.glob("*.txt"):
        stem = p.stem
        if stem not in sizes:
            continue
        w, h = sizes[stem]
        items = []
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = (float(x) for x in parts[1:5])
            x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
            x2, y2 = (xc + bw / 2) * w, (yc + bh / 2) * h
            items.append((cls, np.array([x1, y1, x2, y2], dtype=np.float32), (x2 - x1) * (y2 - y1)))
        gts[stem] = items
    return gts


def _predict(
    model, images: list[Path], device: str, imgsz: int, max_det: int
) -> dict[str, list[tuple[int, float, np.ndarray]]]:
    """Return {stem: [(cls, conf, xyxy)]} via batched inference."""
    results = model.predict(
        source=[str(p) for p in images],
        conf=0.001,
        iou=0.7,
        imgsz=imgsz,
        device=device,
        max_det=max_det,
        verbose=False,
    )
    dets: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    for img, r in zip(images, results):
        items = []
        if r.boxes is not None and len(r.boxes) > 0:
            for cls, conf, xyxy in zip(
                r.boxes.cls.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.xyxy.cpu().numpy(),
            ):
                items.append((int(cls), float(conf), np.asarray(xyxy, dtype=np.float32)))
        dets[img.stem] = items
    return dets


def _match_class(dets_cls, gts_cls_img, image_ids, iou_thr):
    """Greedy per-image match (confidence desc). Returns detections sorted desc as (conf, matched_area|None)."""
    order = sorted(range(len(dets_cls)), key=lambda i: -dets_cls[i][1])
    used = {iid: np.zeros(len(gts_cls_img.get(iid, [])), dtype=bool) for iid in image_ids}
    out = []
    for rank in order:
        iid, conf, box = dets_cls[rank]
        gts = gts_cls_img.get(iid)
        matched_area = None
        if gts:
            boxes = np.stack([g[0] for g in gts])
            ious = _iou(box[None], boxes)[0]
            best = int(ious.argmax())
            if ious[best] >= iou_thr and not used[iid][best]:
                used[iid][best] = True
                matched_area = gts[best][1]
        out.append((conf, matched_area))
    return out


def _ap_for_area(out, gt_areas, lo, hi):
    """AP from pre-matched detections for the area range [lo, hi)."""
    npos = sum(1 for a in gt_areas if lo <= a < (hi if hi is not None else float("inf")))
    if npos == 0:
        return 0.0
    tp, fp = [], []
    for _, area in out:  # out already sorted desc by conf
        if area is None:
            tp.append(0)
            fp.append(1)
        elif lo <= area < (hi if hi is not None else float("inf")):
            tp.append(1)
            fp.append(0)
        # else: matched GT outside tier -> ignored
    tp = np.cumsum(np.asarray(tp, dtype=np.float32))
    fp = np.cumsum(np.asarray(fp, dtype=np.float32))
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-9)
    return _ap_from_pr(rec, prec)


def main() -> None:
    parser = argparse.ArgumentParser(description="VisDrone tiered AP evaluation")
    parser.add_argument("--weights", required=True, help="trained checkpoint, e.g. .../weights/best.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=500)
    args = parser.parse_args()

    data = YAML.load(DATA_YAML)
    root = Path(data["path"])
    images = sorted((root / "images" / "val").glob("*.jpg"))
    sizes = _img_sizes(root / "images" / "val")
    gts = _load_gt(root / "labels" / "val", sizes)

    print(f"Loading {args.weights} and inferring {len(images)} val images...")
    model = YOLO(args.weights)
    dets = _predict(model, images, args.device, args.imgsz, args.max_det)
    image_ids = sorted(set(gts) | set(dets))

    # Group GT per (class, image) and collect per-class GT areas.
    gts_cls_img: dict[int, dict[str, list[tuple[np.ndarray, float]]]] = {}
    gt_areas_cls: dict[int, list[float]] = {}
    for iid, items in gts.items():
        for cls, box, area in items:
            gts_cls_img.setdefault(cls, {}).setdefault(iid, []).append((box, area))
            gt_areas_cls.setdefault(cls, []).append(area)

    dets_cls: dict[int, list[tuple[str, float, np.ndarray]]] = {}
    for iid, items in dets.items():
        for cls, conf, box in items:
            dets_cls.setdefault(cls, []).append((iid, conf, box))

    classes = sorted(set(gt_areas_cls))
    tiers = DEFAULT_TIERS + [("all", 0, None)]

    # Accumulate AP per tier across IoU thresholds.
    tier_aps: dict[str, list[float]] = {label: [] for label, _, _ in tiers}
    tier_counts: dict[str, int] = {label: 0 for label, _, _ in tiers}
    for iou_thr in IOU_THRS:
        per_cls = {label: [] for label, _, _ in tiers}
        for c in classes:
            out = _match_class(dets_cls.get(c, []), gts_cls_img[c], image_ids, iou_thr)
            for label, lo, hi in tiers:
                per_cls[label].append(_ap_for_area(out, gt_areas_cls[c], lo, hi))
        for label, _, _ in tiers:
            tier_aps[label].append(float(np.mean(per_cls[label])))

    for label, lo, hi in tiers:
        upper = hi if hi is not None else float("inf")
        tier_counts[label] = sum(1 for areas in gt_areas_cls.values() for a in areas if lo <= a < upper)

    # Report.
    print("\n=== VisDrone tiered AP (area of matched GT) ===")
    print(f"{'tier':<8}{'instances':>10}{'mAP50':>10}{'mAP50-95':>12}")
    for label, _, _ in tiers:
        map50 = tier_aps[label][0]
        map5095 = float(np.mean(tier_aps[label]))
        print(f"{label:<8}{tier_counts[label]:>10}{map50:>10.4f}{map5095:>12.4f}")


if __name__ == "__main__":
    main()
