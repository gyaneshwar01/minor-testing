"""
Evaluation pipeline for Neplish ASR.

Computes:
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Code-Switch WER (on code-switched subset only)
  - English Token Accuracy (precision / recall for English words)

Usage:
    python -m src.evaluation.evaluate --model_dir models/whisper-neplish/final
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import PeftModel

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def load_model(model_dir: str, base_model_name: str = "openai/whisper-large-v3"):
    """Load a fine-tuned QLoRA Whisper model.

    The base model is loaded in fp16 (not quantized) and the LoRA
    adapter is merged back so that inference runs at full precision.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading processor from %s", model_dir)
    processor = WhisperProcessor.from_pretrained(model_dir)

    logger.info("Loading base model %s + LoRA from %s", base_model_name, model_dir)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_name, torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()

    return model, processor, device


def load_vanilla_model(model_name: str = "openai/whisper-large-v3"):
    """Load vanilla (non-fine-tuned) Whisper for baseline comparison."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(model_name, language="ne", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16,
    )
    model.to(device)
    model.eval()
    return model, processor, device


def transcribe_batch(
    model,
    processor,
    device: str,
    audio_arrays: list,
    sampling_rate: int = 16_000,
    language: str = "ne",
) -> list[str]:
    """Transcribe a list of audio arrays."""
    input_features = processor.feature_extractor(
        audio_arrays,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    ).input_features.to(device=device, dtype=model.dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language, task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    transcriptions = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcriptions


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_wer_cer(predictions: list[str], references: list[str]) -> dict:
    """Compute WER and CER."""
    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)

    return {"wer": round(wer * 100, 2), "cer": round(cer * 100, 2)}


def compute_code_switch_metrics(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute WER only on code-switched samples (those containing English words)."""
    cs_preds, cs_refs = [], []
    non_cs_preds, non_cs_refs = [], []

    for pred, ref in zip(predictions, references):
        if re.search(r"[a-zA-Z]{2,}", ref):
            cs_preds.append(pred)
            cs_refs.append(ref)
        else:
            non_cs_preds.append(pred)
            non_cs_refs.append(ref)

    results = {}

    if cs_refs:
        wer_metric = hf_evaluate.load("wer")
        cs_wer = wer_metric.compute(predictions=cs_preds, references=cs_refs)
        results["cs_wer"] = round(cs_wer * 100, 2)
        results["cs_count"] = len(cs_refs)
    else:
        results["cs_wer"] = None
        results["cs_count"] = 0

    if non_cs_refs:
        wer_metric = hf_evaluate.load("wer")
        non_cs_wer = wer_metric.compute(predictions=non_cs_preds, references=non_cs_refs)
        results["nepali_only_wer"] = round(non_cs_wer * 100, 2)
        results["nepali_only_count"] = len(non_cs_refs)
    else:
        results["nepali_only_wer"] = None
        results["nepali_only_count"] = 0

    return results


def compute_english_token_accuracy(
    predictions: list[str],
    references: list[str],
) -> dict:
    """Compute precision and recall for English tokens in code-switched samples."""
    total_ref_english = 0
    total_pred_english = 0
    correctly_predicted = 0

    for pred, ref in zip(predictions, references):
        ref_eng = set(w.lower() for w in re.findall(r"[a-zA-Z]{2,}", ref))
        pred_eng = set(w.lower() for w in re.findall(r"[a-zA-Z]{2,}", pred))

        total_ref_english += len(ref_eng)
        total_pred_english += len(pred_eng)
        correctly_predicted += len(ref_eng & pred_eng)

    precision = correctly_predicted / total_pred_english if total_pred_english > 0 else 0
    recall = correctly_predicted / total_ref_english if total_ref_english > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "eng_token_precision": round(precision * 100, 2),
        "eng_token_recall": round(recall * 100, 2),
        "eng_token_f1": round(f1 * 100, 2),
        "total_ref_english_tokens": total_ref_english,
        "total_pred_english_tokens": total_pred_english,
        "correctly_predicted_tokens": correctly_predicted,
    }


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    processor,
    device: str,
    dataset,
    batch_size: int = 8,
    label: str = "fine-tuned",
) -> dict:
    """Run evaluation on a dataset split and compute all metrics."""
    predictions = []
    references = []

    logger.info("Running inference (%s) on %d samples ...", label, len(dataset))

    for i in tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating ({label})"):
        batch = dataset[i : i + batch_size]

        audio_arrays = [a["array"] for a in batch["audio"]]
        refs = batch["sentence"]

        preds = transcribe_batch(model, processor, device, audio_arrays)

        predictions.extend(preds)
        references.extend(refs)

    # Compute all metrics
    logger.info("Computing metrics ...")
    overall = compute_wer_cer(predictions, references)
    cs_metrics = compute_code_switch_metrics(predictions, references)
    eng_metrics = compute_english_token_accuracy(predictions, references)

    results = {
        "label": label,
        "num_samples": len(references),
        **overall,
        **cs_metrics,
        **eng_metrics,
    }

    return results, predictions, references


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results: {results['label']}")
    print(f"{'='*60}")
    print(f"  Samples evaluated     : {results['num_samples']}")
    print(f"  Overall WER           : {results['wer']}%")
    print(f"  Overall CER           : {results['cer']}%")
    print(f"  ---")
    print(f"  Code-Switch WER       : {results['cs_wer']}% (n={results['cs_count']})")
    print(f"  Nepali-only WER       : {results['nepali_only_wer']}% (n={results['nepali_only_count']})")
    print(f"  ---")
    print(f"  English Token Prec    : {results['eng_token_precision']}%")
    print(f"  English Token Recall  : {results['eng_token_recall']}%")
    print(f"  English Token F1      : {results['eng_token_f1']}%")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Neplish ASR model.")
    parser.add_argument(
        "--model_dir", type=str,
        default=str(PROJECT_ROOT / "models" / "whisper-neplish" / "final"),
        help="Path to the fine-tuned model directory.",
    )
    parser.add_argument(
        "--base_model", type=str, default="openai/whisper-large-v3",
        help="Base Whisper model name (for QLoRA loading).",
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

    # Evaluate fine-tuned model
    model, processor, device = load_model(args.model_dir, args.base_model)
    results, preds, refs = evaluate_model(
        model, processor, device, test_ds,
        batch_size=args.batch_size, label="fine-tuned",
    )
    print_results(results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        output_path = str(PROJECT_ROOT / "evaluation_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    # Show some example predictions
    print("\nSample Predictions:")
    print("-" * 60)
    for i in range(min(10, len(preds))):
        print(f"  REF : {refs[i][:120]}")
        print(f"  PRED: {preds[i][:120]}")
        print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

