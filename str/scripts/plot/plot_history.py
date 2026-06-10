"""Plot training history curves and optionally write TensorBoard scalars."""

from __future__ import annotations

import argparse
import json
import os
import sys
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


def numeric_metric_columns(history: pd.DataFrame) -> list[str]:
    columns = []
    for column in history.columns:
        if column == "epoch":
            continue
        values = pd.to_numeric(history[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def plot_history(history_csv: Path, output_png: Path, title: str) -> dict[str, object]:
    plt = load_matplotlib()
    history = pd.read_csv(history_csv)
    if "epoch" not in history.columns:
        raise ValueError(f"{history_csv} does not contain an epoch column")

    metric_columns = numeric_metric_columns(history)
    if not metric_columns:
        raise ValueError(f"{history_csv} does not contain numeric metric columns")

    epochs = pd.to_numeric(history["epoch"], errors="coerce")
    fig_width = max(10.0, min(18.0, 1.2 * len(metric_columns)))
    fig, ax = plt.subplots(figsize=(fig_width, 6.5))
    for column in metric_columns:
        values = pd.to_numeric(history[column], errors="coerce")
        ax.plot(epochs, values, marker="o", markersize=3, linewidth=1.8, label=column)

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
        "history_csv": display_path(history_csv),
        "output_png": display_path(output_png),
        "title": title,
        "epochs": int(len(history)),
        "metric_columns": metric_columns,
    }


def write_tensorboard_scalars(history_csv: Path, log_dir: Path, run_name: str) -> dict[str, object]:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        return {
            "tensorboard_status": "skipped",
            "tensorboard_warning": f"tensorboard is unavailable: {exc}. Install it with: python -m pip install tensorboard",
        }

    history = pd.read_csv(history_csv)
    metric_columns = numeric_metric_columns(history)
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    try:
        for row in history.itertuples(index=False):
            epoch = int(getattr(row, "epoch"))
            for column in metric_columns:
                value = pd.to_numeric(pd.Series([getattr(row, column)]), errors="coerce").iloc[0]
                if pd.notna(value):
                    writer.add_scalar(column, float(value), epoch)
    finally:
        writer.flush()
        writer.close()

    return {
        "tensorboard_status": "written",
        "tensorboard_run_dir": display_path(run_dir),
        "metric_columns": metric_columns,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-csv", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument("--tensorboard-run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_csv = project_path(args.history_csv)
    output_png = project_path(args.output_png)
    summary = plot_history(history_csv, output_png, args.title)

    if args.tensorboard_dir:
        run_name = args.tensorboard_run_name or output_png.stem
        tb_summary = write_tensorboard_scalars(history_csv, project_path(args.tensorboard_dir), run_name)
        summary.update(tb_summary)

    if args.summary_json:
        summary_json = project_path(args.summary_json)
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
