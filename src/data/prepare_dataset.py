"""
Prepare HuggingFace Dataset from cleaned CSV for Whisper fine-tuning.

Creates train / validation / test splits stratified by batch and
code-switch presence, then saves them as a HuggingFace DatasetDict
on disk for efficient loading during training.

Usage:
    python -m src.data.prepare_dataset [--output_dir data/hf_dataset]
"""

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "Dataset"


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """Create a composite key for stratified splitting.

    Combines batch identity and whether the sample contains English
    code-switching so that splits are balanced along both axes.
    """
    has_eng = df["sentence"].apply(lambda s: bool(re.search(r"[a-zA-Z]{2,}", str(s))))
    return df["batch"].astype(str) + "_" + has_eng.astype(str)


def prepare_dataframe(csv_path: str | Path) -> pd.DataFrame:
    """Load the cleaned CSV and fix paths to be absolute."""
    df = pd.read_csv(csv_path)

    # Ensure the audio path column points to absolute paths
    def _resolve(rel_path: str) -> str:
        abs_path = DATASET_DIR / rel_path
        return str(abs_path)

    df["audio"] = df["path_relative"].apply(_resolve)

    # Keep only columns needed for training
    df = df[["audio", "sentence", "batch"]].copy()
    return df


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.1,
    val_size: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Split into train / val / test with stratification."""
    strat_key = create_stratification_key(df)

    # First split off test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=strat_key
    )

    # Then split train_val into train and val
    strat_key_tv = create_stratification_key(train_val_df)
    relative_val = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, random_state=seed, stratify=strat_key_tv
    )

    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def build_hf_dataset(splits: dict[str, pd.DataFrame]) -> DatasetDict:
    """Convert pandas DataFrames into a HuggingFace DatasetDict with Audio feature."""
    ds_dict = {}
    for split_name, split_df in splits.items():
        # Build from dict to avoid pyarrow large_string issues
        ds = Dataset.from_dict({
            "audio": split_df["audio"].tolist(),
            "sentence": split_df["sentence"].tolist(),
        })
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
        ds_dict[split_name] = ds
        logger.info(
            "  %s: %d samples", split_name, len(ds)
        )
    return DatasetDict(ds_dict)


def main(output_dir: str | None = None) -> None:
    """End-to-end dataset preparation."""
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "hf_dataset")

    cleaned_csv = DATASET_DIR / "cleaned_dataset.csv"
    if not cleaned_csv.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {cleaned_csv}. "
            "Run `python -m src.data.clean_transcripts` first."
        )

    logger.info("Loading cleaned dataset from %s", cleaned_csv)
    df = prepare_dataframe(cleaned_csv)
    logger.info("Total samples: %d", len(df))

    logger.info("Splitting dataset (80/10/10) ...")
    splits = split_dataset(df)

    for name, sdf in splits.items():
        has_eng = sdf["sentence"].apply(lambda s: bool(re.search(r"[a-zA-Z]{2,}", str(s))))
        logger.info(
            "  %s: %d samples (%d code-switched, %.1f%%)",
            name, len(sdf), has_eng.sum(), has_eng.mean() * 100,
        )

    logger.info("Building HuggingFace DatasetDict ...")
    hf_ds = build_hf_dataset(splits)

    logger.info("Saving to %s ...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    hf_ds.save_to_disk(output_dir)

    print(f"\nDataset saved to {output_dir}")
    print(hf_ds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare HF dataset for Whisper fine-tuning.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save the HuggingFace DatasetDict.",
    )
    args = parser.parse_args()
    main(output_dir=args.output_dir)

