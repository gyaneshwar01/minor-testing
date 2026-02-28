"""
Data Augmentation Pipeline for Neplish ASR Dataset.

Applies audio-level augmentations to the HuggingFace dataset to improve
model robustness. Also oversamples pure-Nepali and code-switched (English)
subsets to balance representation.

Augmentations applied:
  1. Additive noise injection (white noise, pink noise)
  2. Speed perturbation (0.9x - 1.1x)
  3. Pitch shifting (±2 semitones)
  4. Volume perturbation (gain ±6 dB)
  5. Time-domain shift (random roll)
  6. Oversampling of pure-Nepali and code-switched subsets with
     different augmentation combos

Usage:
    python -m src.data.augment_dataset [--input_dir data/hf_dataset]
                                       [--output_dir data/hf_dataset_augmented]
                                       [--augment_factor 2]
"""

import argparse
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Settings for each augmentation type."""

    # Noise injection
    noise_enabled: bool = True
    white_noise_snr_range: tuple[float, float] = (15.0, 30.0)  # dB
    pink_noise_snr_range: tuple[float, float] = (15.0, 30.0)   # dB

    # Speed perturbation
    speed_enabled: bool = True
    speed_range: tuple[float, float] = (0.9, 1.1)

    # Pitch shifting
    pitch_enabled: bool = True
    pitch_semitone_range: tuple[float, float] = (-2.0, 2.0)

    # Volume perturbation
    volume_enabled: bool = True
    volume_gain_db_range: tuple[float, float] = (-6.0, 6.0)

    # Time shift
    time_shift_enabled: bool = True
    time_shift_max_fraction: float = 0.1  # max 10% of audio length

    # Oversampling
    oversample_pure_nepali: bool = True
    oversample_code_switched: bool = True
    nepali_augment_factor: int = 1   # extra copies of pure-Nepali samples
    english_augment_factor: int = 2  # extra copies of code-switched samples

    # General
    augment_factor: int = 2          # how many augmented copies per sample
    seed: int = 42
    sampling_rate: int = 16_000


# ---------------------------------------------------------------------------
# Noise generation utilities
# ---------------------------------------------------------------------------

def _generate_white_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate white (Gaussian) noise."""
    return rng.standard_normal(length).astype(np.float32)


def _generate_pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate pink (1/f) noise using the Voss-McCartney algorithm."""
    # Simplified approach: filter white noise
    white = rng.standard_normal(length).astype(np.float32)
    # Apply a simple 1/f filter via cumulative sum + decay
    b = [0.049922035, -0.095993537, 0.050612699, -0.004709510]
    a = [1.0, -2.494956002, 2.017265875, -0.522189400]

    # Use simple IIR filtering
    from scipy.signal import lfilter
    pink = lfilter(b, a, white).astype(np.float32)

    # Normalise to unit variance
    std = np.std(pink)
    if std > 0:
        pink = pink / std
    return pink


def _add_noise(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """Mix audio with noise at a given SNR (in dB)."""
    audio_power = np.mean(audio ** 2)
    if audio_power < 1e-10:
        return audio

    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return audio

    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return (audio + scale * noise).astype(np.float32)


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def add_white_noise(
    audio: np.ndarray,
    rng: np.random.Generator,
    snr_range: tuple[float, float] = (15.0, 30.0),
) -> np.ndarray:
    """Add white Gaussian noise at a random SNR."""
    snr_db = rng.uniform(*snr_range)
    noise = _generate_white_noise(len(audio), rng)
    return _add_noise(audio, noise, snr_db)


def add_pink_noise(
    audio: np.ndarray,
    rng: np.random.Generator,
    snr_range: tuple[float, float] = (15.0, 30.0),
) -> np.ndarray:
    """Add pink (1/f) noise at a random SNR."""
    snr_db = rng.uniform(*snr_range)
    noise = _generate_pink_noise(len(audio), rng)
    # Ensure noise length matches audio
    if len(noise) > len(audio):
        noise = noise[: len(audio)]
    elif len(noise) < len(audio):
        noise = np.pad(noise, (0, len(audio) - len(noise)))
    return _add_noise(audio, noise, snr_db)


def speed_perturbation(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    speed_range: tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """Apply speed perturbation via resampling (changes tempo and pitch)."""
    import librosa

    speed_factor = rng.uniform(*speed_range)
    # Resample to simulate speed change
    augmented = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=int(sr * speed_factor),
    )
    # Resample back to original SR to keep consistent sample rate
    augmented = librosa.resample(
        augmented,
        orig_sr=int(sr * speed_factor),
        target_sr=sr,
    )
    return augmented.astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    semitone_range: tuple[float, float] = (-2.0, 2.0),
) -> np.ndarray:
    """Shift pitch by a random number of semitones."""
    import librosa

    n_steps = rng.uniform(*semitone_range)
    augmented = librosa.effects.pitch_shift(
        y=audio, sr=sr, n_steps=n_steps
    )
    return augmented.astype(np.float32)


def volume_perturbation(
    audio: np.ndarray,
    rng: np.random.Generator,
    gain_db_range: tuple[float, float] = (-6.0, 6.0),
) -> np.ndarray:
    """Apply random volume gain in dB."""
    gain_db = rng.uniform(*gain_db_range)
    gain_linear = 10 ** (gain_db / 20)
    augmented = audio * gain_linear

    # Clip to prevent clipping
    augmented = np.clip(augmented, -1.0, 1.0)
    return augmented.astype(np.float32)


def time_shift(
    audio: np.ndarray,
    rng: np.random.Generator,
    max_fraction: float = 0.1,
) -> np.ndarray:
    """Randomly shift audio in time (circular roll)."""
    max_shift = int(len(audio) * max_fraction)
    if max_shift == 0:
        return audio
    shift = rng.integers(-max_shift, max_shift)
    return np.roll(audio, shift).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def augment_audio(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    config: AugmentationConfig,
    augment_type: Optional[str] = None,
) -> np.ndarray:
    """Apply a random combination of augmentations to an audio sample.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (float32, mono).
    sr : int
        Sampling rate.
    rng : np.random.Generator
        Random number generator for reproducibility.
    config : AugmentationConfig
        Augmentation settings.
    augment_type : str, optional
        If specified, apply only this augmentation type. One of:
        "white_noise", "pink_noise", "speed", "pitch", "volume", "time_shift",
        "combo" (random combination of 2-3 augmentations).
        If None, a random type is chosen.

    Returns
    -------
    np.ndarray
        Augmented audio waveform.
    """
    available_augmentations = []
    if config.noise_enabled:
        available_augmentations.extend(["white_noise", "pink_noise"])
    if config.speed_enabled:
        available_augmentations.append("speed")
    if config.pitch_enabled:
        available_augmentations.append("pitch")
    if config.volume_enabled:
        available_augmentations.append("volume")
    if config.time_shift_enabled:
        available_augmentations.append("time_shift")

    if not available_augmentations:
        return audio

    if augment_type is None:
        # 50% chance of single augmentation, 50% chance of combo
        if rng.random() < 0.5:
            augment_type = rng.choice(available_augmentations)
        else:
            augment_type = "combo"

    augmented = audio.copy()

    if augment_type == "combo":
        # Apply 2-3 random augmentations in sequence
        n_augs = rng.integers(2, min(4, len(available_augmentations) + 1))
        chosen = rng.choice(available_augmentations, size=n_augs, replace=False)
        for aug in chosen:
            augmented = _apply_single_augmentation(augmented, sr, rng, config, aug)
    else:
        augmented = _apply_single_augmentation(augmented, sr, rng, config, augment_type)

    return augmented


def _apply_single_augmentation(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    config: AugmentationConfig,
    aug_type: str,
) -> np.ndarray:
    """Apply a single named augmentation."""
    if aug_type == "white_noise":
        return add_white_noise(audio, rng, config.white_noise_snr_range)
    elif aug_type == "pink_noise":
        return add_pink_noise(audio, rng, config.pink_noise_snr_range)
    elif aug_type == "speed":
        return speed_perturbation(audio, sr, rng, config.speed_range)
    elif aug_type == "pitch":
        return pitch_shift(audio, sr, rng, config.pitch_semitone_range)
    elif aug_type == "volume":
        return volume_perturbation(audio, rng, config.volume_gain_db_range)
    elif aug_type == "time_shift":
        return time_shift(audio, rng, config.time_shift_max_fraction)
    else:
        logger.warning("Unknown augmentation type: %s", aug_type)
        return audio


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------

def _is_code_switched(sentence: str) -> bool:
    """Check if a sentence contains English (Latin-script) words."""
    return bool(re.search(r"[a-zA-Z]{2,}", str(sentence)))


def augment_single_example(
    example: dict,
    rng: np.random.Generator,
    config: AugmentationConfig,
    augment_type: Optional[str] = None,
) -> dict:
    """Augment a single dataset example (audio + keep sentence unchanged)."""
    audio_array = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    augmented_audio = augment_audio(audio_array, sr, rng, config, augment_type)

    return {
        "audio": {
            "array": augmented_audio,
            "sampling_rate": sr,
            "path": example["audio"].get("path", ""),
        },
        "sentence": example["sentence"],
    }


def augment_dataset(
    dataset: Dataset,
    config: AugmentationConfig,
    split_name: str = "train",
) -> Dataset:
    """Create augmented copies of a dataset split.

    Steps:
      1. Separate into pure-Nepali and code-switched subsets.
      2. Apply augmentations with oversampling factors per subset.
      3. Concatenate original + augmented data.

    Parameters
    ----------
    dataset : Dataset
        A single split (e.g. train) of the HF dataset.
    config : AugmentationConfig
        Augmentation settings.
    split_name : str
        Name of the split (for logging).

    Returns
    -------
    Dataset
        Augmented dataset (original + augmented copies).
    """
    rng = np.random.default_rng(config.seed)
    sentences = dataset["sentence"]

    # Identify subsets
    cs_indices = [i for i, s in enumerate(sentences) if _is_code_switched(s)]
    nepali_indices = [i for i, s in enumerate(sentences) if not _is_code_switched(s)]

    logger.info(
        "[%s] Found %d pure-Nepali and %d code-switched samples",
        split_name, len(nepali_indices), len(cs_indices),
    )

    all_augmented = []

    # --- Augment pure-Nepali samples ---
    nepali_factor = config.nepali_augment_factor
    if config.oversample_pure_nepali and nepali_indices and nepali_factor > 0:
        nepali_subset = dataset.select(nepali_indices)
        logger.info(
            "[%s] Creating %d augmented copies of %d pure-Nepali samples ...",
            split_name, nepali_factor, len(nepali_subset),
        )
        for copy_idx in range(nepali_factor):
            aug_data = {"audio": [], "sentence": []}
            for i in range(len(nepali_subset)):
                example = nepali_subset[i]
                aug_example = augment_single_example(
                    example, rng, config, augment_type=None
                )
                aug_data["audio"].append(aug_example["audio"])
                aug_data["sentence"].append(aug_example["sentence"])

            aug_ds = Dataset.from_dict(aug_data)
            aug_ds = aug_ds.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
            all_augmented.append(aug_ds)
            logger.info(
                "[%s] Pure-Nepali augmented copy %d/%d done (%d samples)",
                split_name, copy_idx + 1, nepali_factor, len(aug_ds),
            )

    # --- Augment code-switched (English) samples ---
    english_factor = config.english_augment_factor
    if config.oversample_code_switched and cs_indices and english_factor > 0:
        cs_subset = dataset.select(cs_indices)
        logger.info(
            "[%s] Creating %d augmented copies of %d code-switched samples ...",
            split_name, english_factor, len(cs_subset),
        )
        for copy_idx in range(english_factor):
            aug_data = {"audio": [], "sentence": []}
            for i in range(len(cs_subset)):
                example = cs_subset[i]
                aug_example = augment_single_example(
                    example, rng, config, augment_type=None
                )
                aug_data["audio"].append(aug_example["audio"])
                aug_data["sentence"].append(aug_example["sentence"])

            aug_ds = Dataset.from_dict(aug_data)
            aug_ds = aug_ds.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
            all_augmented.append(aug_ds)
            logger.info(
                "[%s] Code-switched augmented copy %d/%d done (%d samples)",
                split_name, copy_idx + 1, english_factor, len(aug_ds),
            )

    # --- General augmentation of all samples ---
    general_factor = config.augment_factor
    if general_factor > 0:
        logger.info(
            "[%s] Creating %d general augmented copies of all %d samples ...",
            split_name, general_factor, len(dataset),
        )
        for copy_idx in range(general_factor):
            aug_data = {"audio": [], "sentence": []}
            for i in range(len(dataset)):
                example = dataset[i]
                aug_example = augment_single_example(
                    example, rng, config, augment_type=None
                )
                aug_data["audio"].append(aug_example["audio"])
                aug_data["sentence"].append(aug_example["sentence"])

            aug_ds = Dataset.from_dict(aug_data)
            aug_ds = aug_ds.cast_column("audio", Audio(sampling_rate=config.sampling_rate))
            all_augmented.append(aug_ds)
            logger.info(
                "[%s] General augmented copy %d/%d done (%d samples)",
                split_name, copy_idx + 1, general_factor, len(aug_ds),
            )

    # Concatenate original + all augmented
    if all_augmented:
        final_dataset = concatenate_datasets([dataset] + all_augmented)
        final_dataset = final_dataset.shuffle(seed=config.seed)
        logger.info(
            "[%s] Final dataset: %d samples (original: %d, augmented: %d)",
            split_name, len(final_dataset), len(dataset),
            len(final_dataset) - len(dataset),
        )
    else:
        final_dataset = dataset
        logger.info("[%s] No augmentation applied.", split_name)

    return final_dataset


def augment_hf_dataset(
    input_dir: str,
    output_dir: str,
    config: Optional[AugmentationConfig] = None,
    augment_train_only: bool = True,
) -> DatasetDict:
    """Load an HF DatasetDict, augment the training split, and save.

    Parameters
    ----------
    input_dir : str
        Path to the saved HuggingFace DatasetDict.
    output_dir : str
        Path to save the augmented DatasetDict.
    config : AugmentationConfig, optional
        Augmentation settings. Uses defaults if not provided.
    augment_train_only : bool
        If True, only augment the train split. Validation and test
        remain untouched to keep evaluation fair.

    Returns
    -------
    DatasetDict
        The augmented dataset.
    """
    if config is None:
        config = AugmentationConfig()

    logger.info("Loading dataset from %s ...", input_dir)
    dataset = load_from_disk(input_dir)

    result_splits = {}

    for split_name in dataset:
        if augment_train_only and split_name != "train":
            logger.info("Keeping %s split unchanged.", split_name)
            result_splits[split_name] = dataset[split_name]
        else:
            logger.info("Augmenting %s split ...", split_name)
            result_splits[split_name] = augment_dataset(
                dataset[split_name], config, split_name
            )

    augmented_ds = DatasetDict(result_splits)

    logger.info("Saving augmented dataset to %s ...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    augmented_ds.save_to_disk(output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)
    for split_name, split_ds in augmented_ds.items():
        orig_size = len(dataset[split_name])
        new_size = len(split_ds)
        print(f"  {split_name:12s}: {orig_size:>6d} -> {new_size:>6d} samples "
              f"(+{new_size - orig_size})")
    print("=" * 60)
    print(f"\nSaved to: {output_dir}")

    return augmented_ds


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    augment_factor: int = 2,
    nepali_factor: int = 1,
    english_factor: int = 2,
) -> None:
    """Run the augmentation pipeline from the command line."""
    if input_dir is None:
        input_dir = str(PROJECT_ROOT / "data" / "hf_dataset")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "hf_dataset_augmented")

    config = AugmentationConfig(
        augment_factor=augment_factor,
        nepali_augment_factor=nepali_factor,
        english_augment_factor=english_factor,
    )

    logger.info("Augmentation config:")
    logger.info("  General augment factor : %d", config.augment_factor)
    logger.info("  Pure-Nepali factor     : %d", config.nepali_augment_factor)
    logger.info("  Code-switched factor   : %d", config.english_augment_factor)
    logger.info("  Noise enabled          : %s", config.noise_enabled)
    logger.info("  Speed enabled          : %s", config.speed_enabled)
    logger.info("  Pitch enabled          : %s", config.pitch_enabled)
    logger.info("  Volume enabled         : %s", config.volume_enabled)
    logger.info("  Time-shift enabled     : %s", config.time_shift_enabled)

    augment_hf_dataset(input_dir, output_dir, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Augment the Neplish ASR HuggingFace dataset."
    )
    parser.add_argument(
        "--input_dir", type=str, default=None,
        help="Path to the input HuggingFace DatasetDict.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to save the augmented HuggingFace DatasetDict.",
    )
    parser.add_argument(
        "--augment_factor", type=int, default=2,
        help="Number of augmented copies of each sample (general). Default: 2.",
    )
    parser.add_argument(
        "--nepali_factor", type=int, default=1,
        help="Extra augmented copies for pure-Nepali samples. Default: 1.",
    )
    parser.add_argument(
        "--english_factor", type=int, default=2,
        help="Extra augmented copies for code-switched samples. Default: 2.",
    )
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor,
        nepali_factor=args.nepali_factor,
        english_factor=args.english_factor,
    )
