#!/usr/bin/env python3
"""Draw violin plots of predicted GO similarity for known PPI pairs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable


TASKS = ("bp", "cc", "mf")
TASK_NAMES = {
    "bp": "BP",
    "cc": "CC",
    "mf": "MF",
}
COLORS = {
    "bp": "#4C78A8",
    "cc": "#E45756",
    "mf": "#54A24B",
}
NS_MAP = {
    "biological_process": "bp",
    "cellular_component": "cc",
    "molecular_function": "mf",
}


def SimGO(
    go_set1: set[int],
    go_set2: set[int],
    go_terms: list[dict],
    go_term_dis: dict[str, int],
    go_diff_weight: dict[str, float] | None = None,
) -> float:
    """Calculate GO similarity with the same rule used in model.dataset.SimGO."""
    if go_diff_weight is None:
        go_diff_weight = {"bp": 0.4, "cc": 0.4, "mf": 0.2}

    go_sim = 1e-8
    ns_max = {"bp": 0, "cc": 0, "mf": 0}
    ns_max_cnt = {"bp": 0, "cc": 0, "mf": 0}

    for go in go_set1 & go_set2:
        go_key = str(go)
        if go_key not in go_term_dis:
            continue
        ns = NS_MAP[go_terms[go]["namespace"]]
        if ns_max[ns] < go_term_dis[go_key]:
            ns_max[ns] = go_term_dis[go_key]
            ns_max_cnt[ns] = 1
        elif ns_max[ns] == go_term_dis[go_key]:
            ns_max_cnt[ns] += 1

    for ns in ns_max:
        go_sim += go_diff_weight[ns] * ns_max[ns] * math.sqrt(float(ns_max_cnt[ns]))
    return go_sim


def simgo_to_confidence(simgo: float, scale: float, baseline: float) -> float:
    """Map an unbounded SimGO value to a confidence-like score in [0, 1)."""
    if scale <= 0:
        raise ValueError("--confidence-scale must be greater than 0.")
    adjusted_simgo = max(0.0, simgo - baseline)
    if adjusted_simgo < 1e-7:
        return 0.0
    return adjusted_simgo / (adjusted_simgo + scale)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_go_index(go_id_path: Path) -> dict[str, int]:
    return {go_name: int(go_idx) for go_name, go_idx in load_json(go_id_path).items()}


def convert_predicted_go(
    predicted_go: dict[str, list[str]],
    go_id: dict[str, int],
    task: str,
    go_terms: list[dict],
    go_id_path: Path,
) -> dict[str, set[int]]:
    pid2go: dict[str, set[int]] = {}
    skipped_go = 0

    for pid, go_names in predicted_go.items():
        go_set = set()
        for go_name in go_names:
            go_idx = go_id.get(go_name)
            if go_idx is None:
                skipped_go += 1
                continue
            if NS_MAP[go_terms[go_idx]["namespace"]] != task:
                continue
            go_set.add(go_idx)
        pid2go[pid] = go_set

    if skipped_go:
        print(f"[{task}] skipped {skipped_go} GO terms missing from {go_id_path}")
    return pid2go


def iter_ppi_pairs(ppi_data: Iterable) -> Iterable[tuple[str, str]]:
    for item in ppi_data:
        if isinstance(item, dict):
            u = item.get("u") or item.get("pid1") or item.get("protein1") or item.get("source")
            v = item.get("v") or item.get("pid2") or item.get("protein2") or item.get("target")
        else:
            u, v = item[:2]
        if u and v:
            yield str(u), str(v)


def compute_similarity_distribution(
    pid2go: dict[str, set[int]],
    ppi_pairs: list[tuple[str, str]],
    go_terms: list[dict],
    go_term_dis: dict[str, int],
    task: str,
    score_mode: str,
    confidence_scale: float,
    confidence_baseline: float,
) -> tuple[list[float], int]:
    scores = []
    missing_pairs = 0
    weight = {ns: 0.0 for ns in TASKS}
    weight[task] = 1.0

    for u, v in ppi_pairs:
        if u not in pid2go or v not in pid2go:
            missing_pairs += 1
            continue
        simgo = SimGO(pid2go[u], pid2go[v], go_terms, go_term_dis, weight)
        if score_mode == "confidence":
            scores.append(simgo_to_confidence(simgo, confidence_scale, confidence_baseline))
        else:
            scores.append(simgo)

    return scores, missing_pairs


def compute_jaccard_distribution(
    pid2go: dict[str, set[int]],
    ppi_pairs: list[tuple[str, str]],
) -> tuple[list[float], int]:
    """Calculate GO-set Jaccard similarity for each known PPI pair."""
    scores = []
    missing_pairs = 0
    for protein1, protein2 in ppi_pairs:
        if protein1 not in pid2go or protein2 not in pid2go:
            missing_pairs += 1
            continue
        go_set1 = pid2go[protein1]
        go_set2 = pid2go[protein2]
        union = go_set1 | go_set2
        scores.append(len(go_set1 & go_set2) / len(union) if union else 0.0)
    return scores, missing_pairs


def draw_violin(
    distributions: dict[str, list[float]],
    output_path: Path,
    dpi: int,
    score_label: str,
    is_confidence: bool,
) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = [task for task in TASKS if distributions.get(task)]
    data = [distributions[task] for task in tasks]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    parts = ax.violinplot(data, showmeans=True, showmedians=True, widths=0.8)

    for body, task in zip(parts["bodies"], tasks):
        body.set_facecolor(COLORS[task])
        body.set_edgecolor("#222222")
        body.set_alpha(0.72)

    for key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if key in parts:
            parts[key].set_color("#222222")
            parts[key].set_linewidth(1.2)

    ax.set_xticks(range(1, len(tasks) + 1), [TASK_NAMES[task] for task in tasks])
    ax.set_xlabel("GO ontology")
    ax.set_ylabel(f"Predicted GO {score_label} of known PPI pairs")
    ax.set_title(f"Known PPI predicted GO {score_label} distribution")
    if is_confidence:
        ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for idx, task in enumerate(tasks, start=1):
        values = np.asarray(distributions[task], dtype=float)
        ax.text(
            idx,
            values.max(),
            f"n={len(values)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def kde_density(values: list[float], y_points: list[float]) -> list[float]:
    if not values:
        return [0.0 for _ in y_points]
    if len(values) == 1:
        return [1.0 if y == values[0] else 0.0 for y in y_points]

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    bandwidth = 1.06 * std * (len(values) ** -0.2)
    if bandwidth <= 0:
        bandwidth = max(max(values) - min(values), 1.0) / 30.0

    norm = 1.0 / (len(values) * bandwidth * math.sqrt(2.0 * math.pi))
    densities = []
    for y in y_points:
        density = 0.0
        for value in values:
            z = (y - value) / bandwidth
            density += math.exp(-0.5 * z * z)
        densities.append(norm * density)
    return densities


def draw_violin_svg(
    distributions: dict[str, list[float]],
    output_path: Path,
    score_label: str,
    is_confidence: bool,
) -> Path:
    tasks = [task for task in TASKS if distributions.get(task)]
    all_values = [value for task in tasks for value in distributions[task]]
    if not all_values:
        raise ValueError("No similarity scores were computed.")

    output_path = output_path.with_suffix(".svg")
    width, height = 900, 560
    left, right, top, bottom = 80, 40, 60, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = min(all_values), max(all_values)
    if is_confidence:
        y_min, y_max = 0.0, 1.0
    else:
        y_pad = (y_max - y_min) * 0.08 or 1.0
        y_min -= y_pad
        y_max += y_pad

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    y_ticks = [y_min + (y_max - y_min) * i / 5 for i in range(6)]
    centers = [
        left + plot_width * (idx + 0.5) / len(tasks)
        for idx in range(len(tasks))
    ]
    max_half_width = min(90.0, plot_width / max(len(tasks), 1) * 0.32)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial" font-size="22">Known PPI predicted GO {score_label} distribution</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222" stroke-width="1.5"/>',
    ]

    for tick in y_ticks:
        py = sy(tick)
        svg.extend(
            [
                f'<line x1="{left}" y1="{py:.2f}" x2="{left + plot_width}" y2="{py:.2f}" stroke="#ddd" stroke-width="1"/>',
                f'<text x="{left - 10}" y="{py + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{tick:.2f}</text>',
            ]
        )

    for center, task in zip(centers, tasks):
        values = sorted(distributions[task])
        y_points = [y_min + (y_max - y_min) * i / 120 for i in range(121)]
        densities = kde_density(values, y_points)
        max_density = max(densities) or 1.0

        right_points = [
            (center + density / max_density * max_half_width, sy(y))
            for y, density in zip(y_points, densities)
        ]
        left_points = [
            (center - density / max_density * max_half_width, sy(y))
            for y, density in reversed(list(zip(y_points, densities)))
        ]
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in right_points + left_points)
        median = values[len(values) // 2]
        mean = sum(values) / len(values)

        svg.extend(
            [
                f'<polygon points="{points}" fill="{COLORS[task]}" fill-opacity="0.72" stroke="#222" stroke-width="1.2"/>',
                f'<line x1="{center - max_half_width * 0.55:.2f}" y1="{sy(median):.2f}" x2="{center + max_half_width * 0.55:.2f}" y2="{sy(median):.2f}" stroke="#222" stroke-width="2"/>',
                f'<circle cx="{center:.2f}" cy="{sy(mean):.2f}" r="4" fill="#222"/>',
                f'<text x="{center:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Arial" font-size="15">{TASK_NAMES[task]}</text>',
                f'<text x="{center:.2f}" y="{sy(max(values)) - 8:.2f}" text-anchor="middle" font-family="Arial" font-size="11">n={len(values)}</text>',
            ]
        )

    svg.extend(
        [
            f'<text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="16">GO ontology</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 24 {top + plot_height / 2})">Predicted GO {score_label} of known PPI pairs</text>',
            "</svg>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw BP/CC/MF violin plots of predicted GO similarity for known PPI pairs."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("workspace/7120/model_v1.1/result_7120"),
        help="Directory containing bp_all_result.json, cc_all_result.json and mf_all_result.json.",
    )
    parser.add_argument(
        "--ppi-path",
        type=Path,
        default=Path("raw_data/7120/ppi/ppi_list.json"),
        help="Known PPI list JSON path.",
    )
    parser.add_argument(
        "--go-id-path",
        type=Path,
        default=Path("processed_data/go_id.json"),
        help="GO id mapping JSON path.",
    )
    parser.add_argument(
        "--go-terms-path",
        type=Path,
        default=Path("processed_data/go_terms.json"),
        help="GO terms JSON path.",
    )
    parser.add_argument(
        "--go-term-dis-path",
        type=Path,
        default=Path("dataset/go_term_dis.json"),
        help="GO term depth JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/7120/model_v1.1/result_7120/ppi_pred_go_confidence_violin.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("jaccard", "confidence", "raw"),
        default="jaccard",
        help="Plot GO-set Jaccard scores, confidence-reduced SimGO scores, or raw SimGO values.",
    )
    parser.add_argument(
        "--confidence-scale",
        type=float,
        default=1.0,
        help=(
            "Scale in confidence = max(0, SimGO - baseline) / "
            "(max(0, SimGO - baseline) + scale). Used only with --score-mode confidence."
        ),
    )
    parser.add_argument(
        "--confidence-baseline",
        type=float,
        default=1.0,
        help=(
            "Baseline SimGO value treated as low-confidence shared generic GO signal. "
            "Used only with --score-mode confidence."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    args = parser.parse_args()

    go_id = load_go_index(args.go_id_path)
    go_terms = load_json(args.go_terms_path)
    go_term_dis = load_json(args.go_term_dis_path)
    ppi_pairs = list(iter_ppi_pairs(load_json(args.ppi_path)))

    distributions: dict[str, list[float]] = {}
    score_label = {
        "jaccard": "Jaccard similarity",
        "confidence": "Confidence",
        "raw": "SimGO similarity",
    }[args.score_mode]
    is_confidence = args.score_mode == "confidence"
    for task in TASKS:
        result_path = args.result_dir / f"{task}_all_result.json"
        predicted_go = load_json(result_path)
        pid2go = convert_predicted_go(predicted_go, go_id, task, go_terms, args.go_id_path)
        if args.score_mode == "jaccard":
            scores, missing_pairs = compute_jaccard_distribution(pid2go, ppi_pairs)
        else:
            scores, missing_pairs = compute_similarity_distribution(
                pid2go,
                ppi_pairs,
                go_terms,
                go_term_dis,
                task,
                args.score_mode,
                args.confidence_scale,
                args.confidence_baseline,
            )
        distributions[task] = scores
        print(
            f"{TASK_NAMES[task]}: {len(scores)} PPI pairs plotted, "
            f"{missing_pairs} pairs skipped because a protein is missing from predictions."
        )

    try:
        output_path = draw_violin(
            distributions, args.output, args.dpi, score_label, is_confidence
        )
    except ModuleNotFoundError:
        print("matplotlib/numpy is not installed; saving an SVG figure instead.")
        output_path = draw_violin_svg(distributions, args.output, score_label, is_confidence)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    os.chdir('..')
    main()
