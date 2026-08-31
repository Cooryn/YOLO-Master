#!/usr/bin/env python3
"""Train the P0 VisDrone baseline (yolo11n from scratch) and log per-epoch positive-sample stats.

Registers two callbacks (without editing ``ultralytics/utils/callbacks/``, per project rules):

- ``on_train_epoch_end``: reads the criterion's ``fg_count`` / ``fg_count_by_stride`` and writes
  them into ``trainer.lr`` so they land in ``results.csv`` as ``train/fg_sum`` and
  ``train/fg_s8`` / ``train/fg_s16`` / ``train/fg_s32`` columns (FPN stride = area proxy).
- ``on_fit_epoch_end``: resets the counters. Validation reuses the training criterion (see
  ``engine/validator.py`` ``model.loss(...)``), so the count must be cleared *after* validate to
  avoid polluting the next epoch's total.

Usage:
    python A2/P0/baseline_train.py --epochs 100 --imgsz 640 --batch 16 --device 0   # full baseline
    python A2/P0/baseline_train.py --epochs 2 --imgsz 320 --batch 4                 # quick smoke
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import unwrap_model

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = REPO_ROOT / "datasets" / "VisDrone.yaml"


def _criterion(trainer):
    """Return the native v8DetectionLoss (unwrapping a CompositeCriterion if present)."""
    model = unwrap_model(trainer.model)
    crit = getattr(model, "criterion", None)
    return getattr(crit, "native_criterion", crit) if crit is not None else None


def on_train_epoch_end(trainer) -> None:
    """Write per-epoch positive-sample counts into trainer.lr (survives into results.csv)."""
    crit = _criterion(trainer)
    if crit is None:
        return
    trainer.lr["train/fg_sum"] = int(crit.fg_count)
    for stride, count in sorted(crit.fg_count_by_stride.items()):
        trainer.lr[f"train/fg_s{int(stride)}"] = int(count)


def on_fit_epoch_end(trainer) -> None:
    """Reset counters after validation so the next epoch starts clean."""
    crit = _criterion(trainer)
    if crit is None:
        return
    crit.fg_count = 0
    crit.fg_count_by_stride = {}


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 VisDrone baseline training")
    parser.add_argument("--model", default="yolo11n.yaml", help="model config (from-scratch, no .pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default=str(REPO_ROOT / "runs" / "a2" / "p0"))
    parser.add_argument("--name", default="visdrone-baseline-yolo11n")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        seed=args.seed,
        workers=0,
        pretrained=False,
        project=args.project,
        name=args.name,
    )
    print(f"\nRun artifacts: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()
