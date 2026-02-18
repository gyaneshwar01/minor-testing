"""
Detailed analysis of ASR results with focus on code-switching performance.

Reads evaluation predictions and produces:
  - Per-sample WER breakdown
  - Error categorisation (substitution, deletion, insertion)
  - Code-switch point analysis
  - Comparison charts (fine-tuned vs baseline)

Usage:
    python -m src.evaluation.analyze_results \
        --finetuned_results evaluation_results_finetuned.json \
        --baseline_results evaluation_results_baseline.json
"""

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sns.set_theme(style="whitegrid", palette="muted")


def load_results(path: str) -> dict:
    """Load evaluation results JSON."""
    with open(path) as f:
        return json.load(f)


def compare_models(finetuned: dict, baseline: dict) -> None:
    """Print side-by-side comparison and generate comparison charts."""
    metrics = ["wer", "cer", "cs_wer", "nepali_only_wer", "eng_token_f1"]
    metric_labels = [
        "Overall WER (%)",
        "Overall CER (%)",
        "Code-Switch WER (%)",
        "Nepali-only WER (%)",
        "English Token F1 (%)",
    ]

    print(f"\n{'='*65}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"{'='*65}")

    ft_vals, bl_vals = [], []
    valid_labels = []

    for metric, label in zip(metrics, metric_labels):
        ft_val = finetuned.get(metric)
        bl_val = baseline.get(metric)

        if ft_val is not None and bl_val is not None:
            delta = ft_val - bl_val
            sign = "+" if delta > 0 else ""
            print(f"  {label:<25} {bl_val:>11.2f}% {ft_val:>11.2f}% {sign}{delta:>9.2f}%")
            ft_vals.append(ft_val)
            bl_vals.append(bl_val)
            valid_labels.append(label)
        else:
            print(f"  {label:<25} {'N/A':>12} {'N/A':>12} {'N/A':>10}")

    print(f"{'='*65}\n")

    # Generate comparison bar chart
    if valid_labels:
        x = np.arange(len(valid_labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width / 2, bl_vals, width, label="Baseline (Vanilla Whisper)")
        bars2 = ax.bar(x + width / 2, ft_vals, width, label="Fine-tuned (LoRA)")

        ax.set_ylabel("Score (%)")
        ax.set_title("Model Comparison: Baseline vs Fine-tuned")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, rotation=20, ha="right")
        ax.legend()

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        chart_path = PROJECT_ROOT / "comparison_chart.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Chart saved to {chart_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyse and compare ASR evaluation results.")
    parser.add_argument("--finetuned_results", type=str, required=True)
    parser.add_argument("--baseline_results", type=str, required=True)
    args = parser.parse_args()

    finetuned = load_results(args.finetuned_results)
    baseline = load_results(args.baseline_results)

    compare_models(finetuned, baseline)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

