#!/usr/bin/env python3
"""Draw violin plots of propagated, original GO annotations for known PPI pairs.

The plotting and scoring code is shared with :mod:`draw_violin`; only the GO
label source differs.  Labels in ``protein_info`` are propagated through the
same GO graph traversal used by ``model.dataset.propagate``.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from draw_violin import (
    TASKS,
    TASK_NAMES,
    convert_predicted_go,
    draw_violin,
    draw_violin_svg,
    iter_ppi_pairs,
    load_go_index,
    load_json,
)


def propagate(go_list: list[int], go_graph) -> set[int]:
    """Match ``model.dataset.propagate`` without importing its heavy dependencies."""
    visited: set[int] = set()
    queue = list(go for go in go_list if go in go_graph)
    visited.update(queue)
    while queue:
        node = queue.pop(0)
        for _, child in go_graph.out_edges(node):
            if child not in visited:
                visited.add(child)
                queue.append(child)
    return visited


def load_propagated_go(
    protein_info_path: Path, go_id: dict[str, int], graph_path: Path
) -> dict[str, list[str]]:
    """Load original annotations and return propagated GO IDs by protein."""
    protein_info = load_json(protein_info_path)
    with graph_path.open("rb") as f:
        go_graph = pickle.load(f)

    # ``go_graph_nx`` contains integer vocabulary indices and has no node
    # attributes.  Looking up ``node['id']`` therefore silently produced
    # integers, while ``convert_predicted_go`` expects GO ID strings.
    index_to_go = {index: go_name for go_name, index in go_id.items()}
    propagated: dict[str, list[str]] = {}
    for protein, info in protein_info.items():
        go_indices = [go_id[go] for go in info.get("go", []) if go in go_id]
        propagated[protein] = [index_to_go[node] for node in propagate(go_indices, go_graph)]
    return propagated


def compute_jaccard_distribution(
    pid2go: dict[str, set[int]], ppi_pairs: list[tuple[str, str]]
) -> tuple[list[float], int]:
    """Compute Jaccard similarity for the GO sets of known PPI pairs."""
    scores: list[float] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw BP/CC/MF violin plots from propagated original GO annotations."
    )
    parser.add_argument("--protein-info-path", type=Path, default=Path("processed_data/7120_protein_info.json"))
    parser.add_argument("--ppi-path", type=Path, default=Path("raw_data/7120/ppi/ppi_list.json"))
    parser.add_argument("--go-id-path", type=Path, default=Path("processed_data/go_id.json"))
    parser.add_argument("--go-terms-path", type=Path, default=Path("processed_data/go_terms.json"))
    parser.add_argument("--go-term-dis-path", type=Path, default=Path("dataset/go_term_dis.json"))
    parser.add_argument("--go-graph-path", type=Path, default=Path("dataset/go_graph_nx.pkl"))
    parser.add_argument("--output", type=Path, default=Path("workspace/7120/model_v1.1/result_7120/original_go_confidence_violin.png"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    go_id = load_go_index(args.go_id_path)
    go_terms = load_json(args.go_terms_path)
    ppi_pairs = list(iter_ppi_pairs(load_json(args.ppi_path)))
    propagated_go = load_propagated_go(args.protein_info_path, go_id, args.go_graph_path)

    distributions: dict[str, list[float]] = {}
    for task in TASKS:
        pid2go = convert_predicted_go(propagated_go, go_id, task, go_terms, args.go_id_path)
        scores, missing = compute_jaccard_distribution(pid2go, ppi_pairs)
        distributions[task] = scores
        print(f"{TASK_NAMES[task]}: {len(scores)} PPI pairs plotted, {missing} pairs skipped.")

    label = "Jaccard similarity"
    try:
        output = draw_violin(distributions, args.output, args.dpi, label, False)
    except ModuleNotFoundError:
        print("matplotlib/numpy is not installed; saving an SVG figure instead.")
        output = draw_violin_svg(distributions, args.output, label, False)
    print(f"Saved {output}")


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
