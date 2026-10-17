"""Evaluate the supplied Case 1 checkpoint; run with: python run_demo.py.

This tutorial performs inference, not training. It loads only class definitions
from the original evaluation script, so importing it cannot launch training.
"""
import argparse
import ast
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "PINN_applications" / "Case1_uniform_rainfall_recharge"


def load_network_class():
    """Reuse the original Swish and Net definitions without top-level I/O."""
    source = CASE / "Case1_Error.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name in {"Swish", "Net"}]
    if {node.name for node in classes} != {"Swish", "Net"}:
        raise RuntimeError("The original evaluation network definitions are missing.")
    namespace = {"torch": torch, "nn": nn, "neu_num": 60}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Net"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time", type=int, default=10, choices=range(21),
                        help="Reference time index (0 through 20; default: 10).")
    parser.add_argument("--checkpoint", type=Path,
                        default=CASE / "pth" / "qianxiang_pth" / "final.pth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "demo_output")
    args = parser.parse_args()
    # CPU inference avoids requiring a GPU or matching CUDA installation.
    torch.set_num_threads(2)
    model = load_network_class()()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("module."): value
                           for key, value in state.items()})
    model.eval()

    reference_file = CASE / "comsol-label1.txt"
    reference = np.loadtxt(reference_file, skiprows=1)
    if reference.ndim != 2 or reference.shape[1] != 23:
        raise ValueError("Expected reference columns: x, y, and head at t=0,...,20.")
    coordinates = reference[:, :2]
    target = reference[:, args.time + 2]
    inputs = torch.tensor(coordinates, dtype=torch.float32)
    time = torch.full((len(inputs), 1), float(args.time))
    with torch.inference_mode():
        prediction = model(inputs[:, :1], inputs[:, 1:2], time).numpy().ravel()
    if not np.isfinite(prediction).all():
        raise RuntimeError("Checkpoint produced non-finite predictions.")
    error = np.abs(prediction - target)
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "reference": str(reference_file.resolve()),
        "time": args.time,
        "points": len(prediction),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "relative_l2": float(np.linalg.norm(prediction - target) / np.linalg.norm(target)),
        "python_device": "cpu",
        "torch_version": torch.__version__,
        "note": "Illustrative comparison of the archived checkpoint and reference; not a verified manuscript table reproduction.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "predictions.csv",
               np.column_stack((coordinates, target, prediction, error)),
               delimiter=",", header="x,y,reference_head,predicted_head,absolute_error",
               comments="")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    lower = min(target.min(), prediction.min())
    upper = max(target.max(), prediction.max())
    for ax, values, title in zip(axes, (target, prediction, error),
                                 ("COMSOL reference", "PINN prediction", "Absolute error")):
        options = {"vmin": lower, "vmax": upper} if title != "Absolute error" else {}
        image = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=values,
                           s=3, cmap="viridis", rasterized=True, **options)
        ax.set(xlabel="x", ylabel="y", title=title, xlim=(0, 100), ylim=(0, 100))
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label="Head" if title != "Absolute error" else "Head error")
    fig.suptitle(f"Case 1 archived-checkpoint demonstration, t={args.time}")
    fig.savefig(args.output_dir / "head_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print("Results saved to:", args.output_dir.resolve())


if __name__ == "__main__":
    main()
