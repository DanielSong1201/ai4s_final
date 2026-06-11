"""Plot history.csv from one completed experiment output directory."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "str_matplotlib_cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it with: pip install matplotlib") from exc
    return plt


def metric_columns(history: pd.DataFrame) -> list[str]:
    columns = []
    for column in history.columns:
        if column == "epoch":
            continue
        values = pd.to_numeric(history[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return name.strip("_") or "metric"


def load_history(experiment_dir: Path) -> tuple[Path, pd.DataFrame, list[str]]:
    history_csv = experiment_dir / "history.csv"
    if not history_csv.exists():
        raise FileNotFoundError(f"history.csv not found: {history_csv}")
    history = pd.read_csv(history_csv)
    if "epoch" not in history.columns:
        raise ValueError(f"{history_csv} does not contain an epoch column")
    metrics = metric_columns(history)
    if not metrics:
        raise ValueError(f"{history_csv} does not contain numeric metric columns")
    return history_csv, history, metrics


def plot_combined(experiment_dir: Path, output_png: Path | None, title: str | None) -> dict[str, object]:
    plt = load_matplotlib()
    history_csv, history, metrics = load_history(experiment_dir)
    epochs = pd.to_numeric(history["epoch"], errors="coerce")
    output_png = output_png or experiment_dir / "history_combined.png"
    title = title or f"{experiment_dir.name} history"

    fig, ax = plt.subplots(figsize=(12, 6.5))
    for metric in metrics:
        values = pd.to_numeric(history[metric], errors="coerce")
        ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=metric)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric value")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)

    return {
        "mode": "combined",
        "experiment_dir": display_path(experiment_dir),
        "history_csv": display_path(history_csv),
        "output_png": display_path(output_png),
        "epochs": int(len(history)),
        "metric_columns": metrics,
    }


def plot_separate(experiment_dir: Path, output_dir: Path | None, title_prefix: str | None) -> dict[str, object]:
    plt = load_matplotlib()
    history_csv, history, metrics = load_history(experiment_dir)
    epochs = pd.to_numeric(history["epoch"], errors="coerce")
    output_dir = output_dir or experiment_dir / "history_metric_plots"
    title_prefix = title_prefix or experiment_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    for metric in metrics:
        values = pd.to_numeric(history[metric], errors="coerce")
        output_png = output_dir / f"{safe_filename(metric)}.png"
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=metric)
        ax.set_title(f"{title_prefix} - {metric}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_png, dpi=180)
        plt.close(fig)
        outputs.append(display_path(output_png))

    return {
        "mode": "separate",
        "experiment_dir": display_path(experiment_dir),
        "history_csv": display_path(history_csv),
        "output_dir": display_path(output_dir),
        "output_pngs": outputs,
        "epochs": int(len(history)),
        "metric_columns": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True, help="Directory containing history.csv.")
    parser.add_argument("--mode", choices=["combined", "separate"], required=True)
    parser.add_argument("--output-png", type=Path, help="Combined mode output PNG path.")
    parser.add_argument("--output-dir", type=Path, help="Separate mode output directory.")
    parser.add_argument("--title", help="Combined mode title.")
    parser.add_argument("--title-prefix", help="Separate mode title prefix.")
    parser.add_argument("--summary-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = project_path(args.experiment_dir)
    if args.mode == "combined":
        summary = plot_combined(
            experiment_dir,
            project_path(args.output_png) if args.output_png else None,
            args.title,
        )
    else:
        summary = plot_separate(
            experiment_dir,
            project_path(args.output_dir) if args.output_dir else None,
            args.title_prefix,
        )

    if args.summary_json:
        summary_json = project_path(args.summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
