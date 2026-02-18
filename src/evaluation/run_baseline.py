"""
Run vanilla Whisper baseline on the test set for comparison.

This script evaluates the unmodified Whisper model on the Neplish test set
and saves results that can be compared against the fine-tuned model.

Usage:
    python -m src.evaluation.run_baseline [--model openai/whisper-small]
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_from_disk

from src.evaluation.evaluate import (
    evaluate_model,
    load_vanilla_model,
    print_results,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Run vanilla Whisper baseline evaluation.")
    parser.add_argument(
        "--model", type=str, default="openai/whisper-medium",
        help="Whisper model to use as baseline.",
    )
    parser.add_argument(
        "--dataset_dir", type=str,
        default=str(PROJECT_ROOT / "data" / "hf_dataset"),
        help="Path to HF dataset directory.",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to evaluate on.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save results JSON.",
    )
    args = parser.parse_args()

    # Load dataset
    logger.info("Loading dataset from %s (split=%s)", args.dataset_dir, args.split)
    dataset = load_from_disk(args.dataset_dir)
    test_ds = dataset[args.split]

    # Load vanilla model
    model, processor, device = load_vanilla_model(args.model)

    # Evaluate
    results, preds, refs = evaluate_model(
        model, processor, device, test_ds,
        batch_size=args.batch_size,
        label=f"baseline ({args.model})",
    )
    print_results(results)

    # Save
    output_path = args.output or str(PROJECT_ROOT / "evaluation_results_baseline.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Baseline results saved to %s", output_path)

    # Show some examples
    print("\nSample Baseline Predictions:")
    print("-" * 60)
    for i in range(min(10, len(preds))):
        print(f"  REF : {refs[i][:120]}")
        print(f"  PRED: {preds[i][:120]}")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

