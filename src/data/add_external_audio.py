"""
Download and integrate external pure-Nepali and English audio data.

Adds small amounts of pure-language data to improve the model's ability
to handle both languages individually, which in turn helps code-switching.

Supported sources (via HuggingFace Hub):
  - **Nepali**:  google/fleurs  (ne_np split) -- high-quality read speech
  - **English**: google/fleurs  (en_us split) -- matched domain / style

Both use the same FLEURS dataset for consistency in recording style,
and the Whisper tokenizer already supports both languages.

Usage:
    python -m src.data.add_external_audio [--max_nepali 500]
                                           [--max_english 300]
                                           [--output_dir data/external]

NOTE: The downloaded data is saved as a HuggingFace DatasetDict with
      'nepali' and 'english' splits so it can be merged into the main
      augmented training set by prepare_dataset.py.
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from datasets import Audio, Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Whisper label constraints
# ---------------------------------------------------------------------------
# Whisper tokenizer has a max target length of 448 tokens.  For safety we
# cap transcript character length so tokenization stays well within limits.
MAX_TRANSCRIPT_CHARS = 400   # conservative; ~200 tokens for English
MAX_AUDIO_DURATION_S = 30.0  # Whisper's hard limit


def _filter_sample(sample: dict, max_duration_s: float = MAX_AUDIO_DURATION_S) -> bool:
    """Return True if the sample is usable for training."""
    audio = sample.get("audio")
    text = sample.get("sentence", sample.get("transcription", ""))

    if not text or not text.strip():
        return False
    if len(text) > MAX_TRANSCRIPT_CHARS:
        return False

    # Check audio duration
    if audio is not None:
        array = audio.get("array")
        sr = audio.get("sampling_rate", 16_000)
        if array is not None:
            duration = len(array) / sr
            if duration < 0.5 or duration > max_duration_s:
                return False

    return True


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_fleurs_nepali(
    max_samples: int = 500,
    seed: int = 42,
) -> Dataset:
    """Download Nepali speech from Google FLEURS.

    FLEURS contains ~12 hours of read speech per language with sentence-
    level transcriptions -- a good match for Whisper training.
    """
    logger.info("Downloading FLEURS Nepali (ne_np) ...")
    ds = load_dataset("google/fleurs", "ne_np", split="train", trust_remote_code=True)

    # Rename to match our schema
    ds = ds.rename_column("transcription", "sentence")

    # Resample to 16 kHz (FLEURS is already 16 kHz but ensure it)
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Filter bad samples
    ds = ds.filter(_filter_sample)

    # Sub-sample if we have more than needed
    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("  Sub-sampled to %d Nepali samples.", max_samples)
    else:
        logger.info("  Got %d Nepali samples (requested %d).", len(ds), max_samples)

    # Keep only the columns we need
    keep_cols = {"audio", "sentence"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)

    return ds


def download_fleurs_english(
    max_samples: int = 300,
    seed: int = 42,
) -> Dataset:
    """Download English speech from Google FLEURS.

    We use fewer English samples because:
      - The primary language is Nepali; English only appears in CS contexts.
      - We don't want to bias the model towards English-dominant output.
      - This acts as a regulariser, not a full English training set.
    """
    logger.info("Downloading FLEURS English (en_us) ...")
    ds = load_dataset("google/fleurs", "en_us", split="train", trust_remote_code=True)

    ds = ds.rename_column("transcription", "sentence")
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
    ds = ds.filter(_filter_sample)

    if len(ds) > max_samples:
        ds = ds.shuffle(seed=seed).select(range(max_samples))
        logger.info("  Sub-sampled to %d English samples.", max_samples)
    else:
        logger.info("  Got %d English samples (requested %d).", len(ds), max_samples)

    keep_cols = {"audio", "sentence"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds = ds.remove_columns(remove_cols)

    return ds


def download_common_voice_nepali(
    max_samples: int = 500,
    seed: int = 42,
) -> Dataset | None:
    """Download Nepali speech from Mozilla Common Voice (fallback).

    Common Voice requires agreeing to the license, so this may fail
    if the user hasn't configured their HuggingFace token.  In that case
    we silently return None and rely on FLEURS only.
    """
    try:
        logger.info("Attempting to download Common Voice Nepali (ne) ...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "ne",
            split="train",
            trust_remote_code=True,
        )

        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        ds = ds.filter(_filter_sample)

        if len(ds) > max_samples:
            ds = ds.shuffle(seed=seed).select(range(max_samples))

        keep_cols = {"audio", "sentence"}
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        ds = ds.remove_columns(remove_cols)

        logger.info("  Got %d Common Voice Nepali samples.", len(ds))
        return ds

    except Exception as e:
        logger.warning(
            "Could not download Common Voice Nepali (may need HF token): %s", e
        )
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def download_external_data(
    output_dir: str | None = None,
    max_nepali: int = 500,
    max_english: int = 300,
    include_common_voice: bool = False,
    seed: int = 42,
) -> DatasetDict:
    """Download external audio and save as a HuggingFace DatasetDict.

    Returns a DatasetDict with 'nepali' and 'english' splits, each
    having 'audio' and 'sentence' columns matching the main dataset schema.
    """
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "external")

    # --- Nepali ---
    nepali_ds = download_fleurs_nepali(max_samples=max_nepali, seed=seed)

    if include_common_voice:
        cv_ds = download_common_voice_nepali(max_samples=max_nepali, seed=seed)
        if cv_ds is not None:
            from datasets import concatenate_datasets
            nepali_ds = concatenate_datasets([nepali_ds, cv_ds])
            nepali_ds = nepali_ds.shuffle(seed=seed)
            if len(nepali_ds) > max_nepali:
                nepali_ds = nepali_ds.select(range(max_nepali))
            logger.info("  Combined Nepali: %d samples", len(nepali_ds))

    # --- English ---
    english_ds = download_fleurs_english(max_samples=max_english, seed=seed)

    # --- Build DatasetDict ---
    external = DatasetDict({
        "nepali": nepali_ds,
        "english": english_ds,
    })

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    external.save_to_disk(output_dir)
    logger.info("External data saved to %s", output_dir)

    print(f"\nExternal dataset saved to {output_dir}")
    print(f"  Nepali:  {len(nepali_ds)} samples")
    print(f"  English: {len(english_ds)} samples")

    return external


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download external Nepali & English audio for Neplish ASR."
    )
    parser.add_argument(
        "--max_nepali", type=int, default=500,
        help="Maximum number of pure Nepali samples to download.",
    )
    parser.add_argument(
        "--max_english", type=int, default=300,
        help="Maximum number of pure English samples to download.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save the external dataset.",
    )
    parser.add_argument(
        "--include_common_voice", action="store_true",
        help="Also try downloading from Mozilla Common Voice (needs HF token).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed.",
    )
    args = parser.parse_args()

    download_external_data(
        output_dir=args.output_dir,
        max_nepali=args.max_nepali,
        max_english=args.max_english,
        include_common_voice=args.include_common_voice,
        seed=args.seed,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()
