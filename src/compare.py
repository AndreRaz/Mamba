"""Compare trained segmentation runs from their metrics.json files.

Usage examples:

    python src/compare.py                          # auto-discover runs under outputs/
    python src/compare.py --runs outputs/unet-full outputs/mamba-full
    python src/compare.py --plot --markdown outputs/comparison/report.md
"""

import argparse
import json
from pathlib import Path


TABLE_COLUMNS = [
    ("run", "Run", "{}"),
    ("model", "Model", "{}"),
    ("parameter_count", "Params", "{:,}"),
    ("epochs_run", "Epochs", "{}"),
    ("duration_seconds", "Time (s)", "{:.1f}"),
    ("val_loss", "Val Loss", "{:.4f}"),
    ("val_accuracy", "Val Acc", "{:.4f}"),
    ("dice", "Dice", "{:.4f}"),
    ("iou", "IoU", "{:.4f}"),
    ("precision", "Precision", "{:.4f}"),
    ("recall", "Recall", "{:.4f}"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare trained model runs for the academic report.")
    parser.add_argument("--outputs-dir", default="outputs", help="Directory scanned for run subdirectories.")
    parser.add_argument("--runs", nargs="+", default=None, help="Explicit run directories (each must contain metrics.json).")
    parser.add_argument("--sort-by", default="dice", choices=("dice", "iou", "val_loss", "val_accuracy", "parameter_count", "duration_seconds"))
    parser.add_argument("--plot", action="store_true", help="Generate overlay plots of validation curves.")
    parser.add_argument("--plot-dir", default="outputs/comparison", help="Directory where comparison plots are saved.")
    parser.add_argument("--markdown", default=None, help="Optional path to export the comparison as a Markdown report.")
    return parser.parse_args()


def discover_runs(outputs_dir: str) -> list[Path]:
    root = Path(outputs_dir)
    if not root.is_dir():
        raise SystemExit(f"Outputs directory not found: {root}")
    return sorted(path.parent for path in root.glob("*/metrics.json"))


def load_run(run_dir: Path) -> dict:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise SystemExit(f"No metrics.json in {run_dir}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    history = metrics.get("history", {})
    final = metrics.get("final_metrics", {})
    segmentation = metrics.get("segmentation_metrics", {})

    return {
        "run": run_dir.name,
        "model": metrics.get("model", "?"),
        "parameter_count": metrics.get("parameter_count"),
        "epochs_run": len(history.get("loss", [])),
        "duration_seconds": metrics.get("duration_seconds"),
        "val_loss": final.get("val_loss"),
        "val_accuracy": final.get("val_accuracy"),
        "dice": segmentation.get("dice"),
        "iou": segmentation.get("iou"),
        "precision": segmentation.get("precision"),
        "recall": segmentation.get("recall"),
        "history": history,
    }


def format_cell(value, fmt: str) -> str:
    if value is None:
        return "—"
    return fmt.format(value)


def build_rows(runs: list[dict]) -> list[list[str]]:
    return [
        [format_cell(run.get(key), fmt) for key, _, fmt in TABLE_COLUMNS]
        for run in runs
    ]


def print_table(runs: list[dict]) -> None:
    headers = [header for _, header, _ in TABLE_COLUMNS]
    rows = build_rows(runs)
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def line(cells):
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(line(headers))
    print(line(["-" * width for width in widths]))
    for row in rows:
        print(line(row))


def print_summary(runs: list[dict]) -> None:
    scored = [run for run in runs if run.get("dice") is not None]
    if len(scored) < 2:
        return

    best = scored[0]
    baseline = scored[-1]
    dice_gap = best["dice"] - baseline["dice"]
    print()
    print(f"Best Dice: {best['run']} ({best['dice']:.4f}), "
          f"+{dice_gap:.4f} over {baseline['run']} ({baseline['dice']:.4f}).")
    if best.get("parameter_count") and baseline.get("parameter_count"):
        param_ratio = best["parameter_count"] / baseline["parameter_count"]
        print(f"Parameter ratio ({best['run']} / {baseline['run']}): {param_ratio:.2f}x")
    if best.get("duration_seconds") and baseline.get("duration_seconds"):
        time_ratio = best["duration_seconds"] / baseline["duration_seconds"]
        print(f"Training time ratio: {time_ratio:.2f}x")


def export_markdown(runs: list[dict], path: str) -> None:
    headers = [header for _, header, _ in TABLE_COLUMNS]
    rows = build_rows(runs)
    lines = [
        "# Model Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    lines.append("")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown report saved to: {output}")


def plot_comparison(runs: list[dict], plot_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for metric, title, filename in (
        ("val_loss", "Validation Loss", "compare_val_loss.png"),
        ("val_accuracy", "Validation Accuracy", "compare_val_accuracy.png"),
    ):
        fig, axis = plt.subplots(figsize=(10, 5))
        plotted = False
        for run in runs:
            values = run["history"].get(metric)
            if not values:
                continue
            axis.plot(range(1, len(values) + 1), values, marker="o", markersize=3, label=run["run"])
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.legend()
        axis.grid(True, linestyle="--", alpha=0.6)
        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {plot_path}")


def main():
    args = parse_args()
    run_dirs = [Path(run) for run in args.runs] if args.runs else discover_runs(args.outputs_dir)
    if not run_dirs:
        raise SystemExit("No runs with metrics.json found.")

    runs = [load_run(run_dir) for run_dir in run_dirs]
    ascending = args.sort_by in ("val_loss", "parameter_count", "duration_seconds")
    runs.sort(
        key=lambda run: (run.get(args.sort_by) is None, run.get(args.sort_by) or 0),
        reverse=not ascending,
    )
    # Entries without the sort metric always go last
    runs.sort(key=lambda run: run.get(args.sort_by) is None)

    print_table(runs)
    print_summary(runs)

    if args.markdown:
        export_markdown(runs, args.markdown)
    if args.plot:
        plot_comparison(runs, args.plot_dir)


if __name__ == "__main__":
    main()
