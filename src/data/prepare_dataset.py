"""
Prepare HuggingFace Dataset from cleaned CSV for Whisper fine-tuning.

Creates train / validation / test splits stratified by batch and
code-switch presence, then saves them as a HuggingFace DatasetDict
on disk for efficient loading during training.

Optionally:
  - Merges external pure-Nepali / English audio into the training split.
  - Applies audio augmentation to expand the training set.

Usage:
    # Basic: just create splits (no augmentation)
    python -m src.data.prepare_dataset

    # Full pipeline: splits + external data + augmentation
    python -m src.data.prepare_dataset --augment --add_external
"""

import argparse
import logging
import os
import re
from pathlib import Path

import pandas as pd
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets
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


def merge_external_data(
    hf_ds: DatasetDict,
    external_dir: str | None = None,
    max_nepali: int = 500,
    max_english: int = 300,
    seed: int = 42,
) -> DatasetDict:
    """Merge external pure-Nepali and English audio into the training split.

    External data is ONLY added to 'train'; validation and test remain
    untouched so evaluation metrics stay comparable.
    """
    if external_dir is None:
        external_dir = str(PROJECT_ROOT / "data" / "external")

    ext_path = Path(external_dir)

    # Download external data if not already cached
    if not ext_path.exists():
        logger.info("External data not found at %s -- downloading ...", external_dir)
        from src.data.add_external_audio import download_external_data
        download_external_data(
            output_dir=external_dir,
            max_nepali=max_nepali,
            max_english=max_english,
            seed=seed,
        )

    from datasets import load_from_disk
    external = load_from_disk(external_dir)

    extra_datasets = []

    if "nepali" in external and len(external["nepali"]) > 0:
        logger.info("  Adding %d pure Nepali samples to training set.", len(external["nepali"]))
        extra_datasets.append(external["nepali"])

    if "english" in external and len(external["english"]) > 0:
        logger.info("  Adding %d pure English samples to training set.", len(external["english"]))
        extra_datasets.append(external["english"])

    if extra_datasets:
        original_train = hf_ds["train"]
        all_train = concatenate_datasets([original_train] + extra_datasets)
        all_train = all_train.shuffle(seed=seed)
        hf_ds["train"] = all_train
        logger.info(
            "  Training set after external merge: %d samples (was %d)",
            len(all_train), len(original_train),
        )

    return hf_ds


def main(
    output_dir: str | None = None,
    add_external: bool = False,
    augment: bool = False,
    num_augmented_copies: int = 2,
    augmented_output_dir: str | None = None,
    seed: int = 42,
) -> None:
    """End-to-end dataset preparation."""
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "hf_dataset")
    if augmented_output_dir is None:
        augmented_output_dir = str(PROJECT_ROOT / "data" / "hf_dataset_augmented")

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
    splits = split_dataset(df, seed=seed)

    for name, sdf in splits.items():
        has_eng = sdf["sentence"].apply(lambda s: bool(re.search(r"[a-zA-Z]{2,}", str(s))))
        logger.info(
            "  %s: %d samples (%d code-switched, %.1f%%)",
            name, len(sdf), has_eng.sum(), has_eng.mean() * 100,
        )

    logger.info("Building HuggingFace DatasetDict ...")
    hf_ds = build_hf_dataset(splits)

    # --- Save base (non-augmented) dataset ---
    logger.info("Saving base dataset to %s ...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    hf_ds.save_to_disk(output_dir)
    print(f"\nBase dataset saved to {output_dir}")
    print(hf_ds)

    # --- Optionally merge external pure-language data ---
    if add_external:
        logger.info("Merging external Nepali + English data ...")
        hf_ds = merge_external_data(hf_ds, seed=seed)

    # --- Optionally apply augmentation ---
    if augment:
        logger.info("Running audio augmentation (%d copies) ...", num_augmented_copies)
        from src.data.augment import augment_dataset
        hf_ds = augment_dataset(
            hf_ds,
            num_augmented_copies=num_augmented_copies,
            seed=seed,
        )

        logger.info("Saving augmented dataset to %s ...", augmented_output_dir)
        os.makedirs(augmented_output_dir, exist_ok=True)
        hf_ds.save_to_disk(augmented_output_dir)
        print(f"\nAugmented dataset saved to {augmented_output_dir}")
        print(hf_ds)
    else:
        print("\nSkipping augmentation (use --augment to enable).")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Prepare HF dataset for Whisper fine-tuning.")
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save the base HuggingFace DatasetDict.",
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Apply audio augmentation to the training split.",
    )
    parser.add_argument(
        "--add_external", action="store_true",
        help="Download and merge external Nepali + English audio into training.",
    )
    parser.add_argument(
        "--num_augmented_copies", type=int, default=2,
        help="Number of augmented copies per training sample (default: 2).",
    )
    parser.add_argument(
        "--augmented_output_dir", type=str, default=None,
        help="Directory to save the augmented dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    main(
        output_dir=args.output_dir,
        add_external=args.add_external,
        augment=args.augment,
        num_augmented_copies=args.num_augmented_copies,
        augmented_output_dir=args.augmented_output_dir,
        seed=args.seed,
    )

