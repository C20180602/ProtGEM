#!/usr/bin/env python3
"""Draw parameter sensitivity curves for BP/CC/MF metrics."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_LOG_DIR = Path("workspace/AF2DB/model_v1.1/log")
TASKS = ("bp", "cc", "mf")
METRICS = {
    "auc": "test_auc",
    "aupr": "test_aupr",
    "fmax": "test_fmax",
}
COLORS = {
    "bp": "#1f77b4",
    "cc": "#d62728",
    "mf": "#2ca02c",
}


def parse_log(log_path: Path) -> list[dict[str, float]]:
    pattern = re.compile(
        r"go_min_count=(?P<go_min_count>\d+),\s*"
        r"test_auc=(?P<test_auc>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
        r"test_aupr=(?P<test_aupr>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?),\s*"
        r"test_fmax=(?P<test_fmax>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
    )

    rows: list[dict[str, float]] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            match = pattern.search(line)
            if not match:
                continue
            rows.append(
                {
                    "go_min_count": int(match.group("go_min_count")),
                    "test_auc": float(match.group("test_auc")),
                    "test_aupr": float(match.group("test_aupr")),
                    "test_fmax": float(match.group("test_fmax")),
                }
            )

    if not rows:
        raise ValueError(f"No valid metric rows found in {log_path}")

    return sorted(rows, key=lambda item: item["go_min_count"])


def load_all(log_dir: Path) -> dict[str, list[dict[str, float]]]:
    data = {}
    for task in TASKS:
        log_path = log_dir / f"param_sensitivity_simple_{task}.log"
        if not log_path.exists():
            raise FileNotFoundError(f"Missing log file: {log_path}")
        data[task] = parse_log(log_path)
    return data


def draw_metric(
    data: dict[str, list[dict[str, float]]],
    metric: str,
    metric_key: str,
    output_dir: Path,
    dpi: int,
) -> Path:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))

    for task in TASKS:
        rows = data[task]
        x = [row["go_min_count"] for row in rows]
        y = [row[metric_key] for row in rows]
        plt.plot(
            x,
            y,
            marker="o",
            linewidth=2,
            markersize=4,
            label=task.upper(),
            color=COLORS[task],
        )

    plt.xlabel("go_min_count")
    plt.ylabel(metric.upper())
    plt.title(f"Parameter sensitivity of {metric.upper()}")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"param_sensitivity_{metric}.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return output_path


def draw_metric_svg(
    data: dict[str, list[dict[str, float]]],
    metric: str,
    metric_key: str,
    output_dir: Path,
) -> Path:
    width, height = 900, 560
    left, right, top, bottom = 90, 35, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_x = [row["go_min_count"] for rows in data.values() for row in rows]
    all_y = [row[metric_key] for rows in data.values() for row in rows]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_padding = (y_max - y_min) * 0.08 or 0.01
    y_min = max(0.0, y_min - y_padding)
    y_max = min(1.0, y_max + y_padding)

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="22">Parameter sensitivity of {metric.upper()}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
    ]

    x_ticks = sorted(set(all_x))
    for x in x_ticks:
        px = sx(x)
        svg.extend(
            [
                f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{top + plot_height}" stroke="#ddd" stroke-width="1"/>',
                f'<line x1="{px:.2f}" y1="{top + plot_height}" x2="{px:.2f}" y2="{top + plot_height + 5}" stroke="#222" stroke-width="1"/>',
                f'<text x="{px:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial" font-size="12">{x}</text>',
            ]
        )

    for i in range(6):
        y = y_min + (y_max - y_min) * i / 5
        py = sy(y)
        svg.extend(
            [
                f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_width}" y2="{py:.2f}" stroke="#ddd" stroke-width="1"/>',
                f'<line x1="{left - 5}" y1="{py:.2f}" x2="{left}" y2="{py:.2f}" stroke="#222" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{y:.3f}</text>',
            ]
        )

    for legend_i, task in enumerate(TASKS):
        rows = data[task]
        points = " ".join(f"{sx(row['go_min_count']):.2f},{sy(row[metric_key]):.2f}" for row in rows)
        svg.append(
            f'<polyline fill="none" stroke="{COLORS[task]}" stroke-width="2.5" points="{points}"/>'
        )
        for row in rows:
            svg.append(
                f'<circle cx="{sx(row["go_min_count"]):.2f}" cy="{sy(row[metric_key]):.2f}" r="4" fill="{COLORS[task]}"/>'
            )
        legend_x = width - 160
        legend_y = top + legend_i * 28
        svg.extend(
            [
                f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{COLORS[task]}" stroke-width="2.5"/>',
                f'<text x="{legend_x + 38}" y="{legend_y + 5}" font-family="Arial" font-size="15">{task.upper()}</text>',
            ]
        )

    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="16">go_min_count</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 24 {top + plot_height / 2})">{metric.upper()}</text>',
            "</svg>",
        ]
    )

    output_path = output_dir / f"param_sensitivity_{metric}.svg"
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw AUC/AUPR/Fmax curves from param_sensitivity_simple logs."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing param_sensitivity_simple logs. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory to save figures. Default: {DEFAULT_LOG_DIR}",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(args.log_dir)

    try:
        import matplotlib.pyplot  # noqa: F401

        draw = lambda metric, metric_key: draw_metric(  # noqa: E731
            data, metric, metric_key, args.output_dir, args.dpi
        )
    except ModuleNotFoundError:
        print("matplotlib is not installed; saving SVG figures instead.")
        draw = lambda metric, metric_key: draw_metric_svg(  # noqa: E731
            data, metric, metric_key, args.output_dir
        )

    for metric, metric_key in METRICS.items():
        output_path = draw(metric, metric_key)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')
    main()
