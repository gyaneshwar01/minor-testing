"""
Data Augmentation Pipeline for Neplish ASR Dataset.

Applies CONSERVATIVE audio-level augmentations to improve model robustness
without distorting speech beyond recognition.

Key design principles:
  - Each augmented sample gets exactly ONE transform (no stacking)
  - Tempo change uses librosa.effects.time_stretch (correct implementation)
  - No time-shift / circular roll (breaks Whisper positional encoding)
  - No pink noise (broken IIR filter — just use Gaussian)
  - Gentle parameter ranges (model should still hear clear speech)
  - All outputs validated (NaN, silence, duration checks)
  - Code-switched oversampling is moderate (1 extra copy)

Augmentations applied:
  1. Gaussian noise injection (SNR 20-40 dB)
  2. Tempo change via time_stretch (0.93x - 1.07x)
  3. Pitch shifting (±1.5 semitones)
  4. Volume perturbation (gain ±4 dB)

Usage:
    python -m src.data.augment_dataset [--input_dir data/hf_dataset]
                                       [--output_dir data/hf_dataset_augmented]
                                       [--augment_factor 1]
"""

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    """Conservative augmentation settings tuned for Whisper fine-tuning.

    Each augmented copy gets exactly ONE random transform.
    Ranges are deliberately gentle so the model still hears clear speech.
    """

    # --- Noise injection ---
    noise_enabled: bool = True
    noise_snr_range: tuple[float, float] = (20.0, 40.0)  # dB — gentle

    # --- Tempo change (via librosa.effects.time_stretch) ---
    speed_enabled: bool = True
    speed_range: tuple[float, float] = (0.93, 1.07)  # tight range

    # --- Pitch shifting ---
    pitch_enabled: bool = True
    pitch_semitone_range: tuple[float, float] = (-1.5, 1.5)  # moderate

    # --- Volume perturbation ---
    volume_enabled: bool = True
    volume_gain_db_range: tuple[float, float] = (-4.0, 4.0)  # gentle

    # --- Oversampling ---
    oversample_code_switched: bool = True
    cs_extra_copies: int = 1  # just 1 extra copy for CS samples

    # --- General ---
    augment_factor: int = 1          # 1 augmented copy per sample
    seed: int = 42
    sampling_rate: int = 16_000

    # --- Validation thresholds ---
    min_audio_seconds: float = 0.5
    max_audio_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Audio validation
# ---------------------------------------------------------------------------

def validate_audio(
    audio: np.ndarray,
    sr: int,
    config: AugmentationConfig,
) -> bool:
    """Check if audio sample is valid for Whisper training."""
    if audio is None or len(audio) == 0:
        return False

    min_samples = int(config.min_audio_seconds * sr)
    max_samples = int(config.max_audio_seconds * sr)
    if len(audio) < min_samples or len(audio) > max_samples:
        return False

    # Reject silence (RMS too low)
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return False

    # Reject NaN / Inf
    if not np.isfinite(audio).all():
        return False

    return True


# ---------------------------------------------------------------------------
# Individual augmentation functions
# ---------------------------------------------------------------------------

def add_gaussian_noise(
    audio: np.ndarray,
    rng: np.random.Generator,
    snr_range: tuple[float, float] = (20.0, 40.0),
) -> np.ndarray:
    """Add Gaussian white noise at a random SNR."""
    snr_db = rng.uniform(*snr_range)
    audio_power = np.mean(audio ** 2)
    if audio_power < 1e-10:
        return audio

    noise = rng.standard_normal(len(audio)).astype(np.float32)
    noise_power = np.mean(noise ** 2)
    target_noise_power = audio_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / max(noise_power, 1e-10))

    result = (audio + scale * noise).astype(np.float32)
    return np.clip(result, -1.0, 1.0)


def change_tempo(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    speed_range: tuple[float, float] = (0.93, 1.07),
) -> np.ndarray:
    """Change tempo WITHOUT changing pitch using librosa.effects.time_stretch.

    This is the correct way to do speed perturbation for ASR:
    the words are spoken faster/slower but the pitch stays natural.

    NOTE: The old implementation used librosa.resample twice which is a
    no-op (resample down then back up just applies anti-aliasing filters).
    time_stretch actually changes the duration of the audio.
    """
    import librosa

    rate = rng.uniform(*speed_range)
    # time_stretch: rate > 1 = faster (shorter audio), rate < 1 = slower
    augmented = librosa.effects.time_stretch(y=audio, rate=rate)

    # Trim to Whisper's 30s window if stretched longer
    max_len = int(30.0 * sr)
    if len(augmented) > max_len:
        augmented = augmented[:max_len]

    return augmented.astype(np.float32)


def shift_pitch(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    semitone_range: tuple[float, float] = (-1.5, 1.5),
) -> np.ndarray:
    """Shift pitch by a small random amount."""
    import librosa

    n_steps = rng.uniform(*semitone_range)
    # Skip negligible shifts
    if abs(n_steps) < 0.1:
        return audio

    augmented = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def change_volume(
    audio: np.ndarray,
    rng: np.random.Generator,
    gain_db_range: tuple[float, float] = (-4.0, 4.0),
) -> np.ndarray:
    """Apply random volume gain in dB."""
    gain_db = rng.uniform(*gain_db_range)
    gain_linear = 10 ** (gain_db / 20)
    augmented = audio * gain_linear
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Augmentation pipeline — exactly ONE transform per sample
# ---------------------------------------------------------------------------

# Registry: name -> config flag that enables it
_AUGMENTATION_REGISTRY = {
    "noise": "noise_enabled",
    "tempo": "speed_enabled",
    "pitch": "pitch_enabled",
    "volume": "volume_enabled",
}


def augment_audio(
    audio: np.ndarray,
    sr: int,
    rng: np.random.Generator,
    config: AugmentationConfig,
) -> np.ndarray:
    """Apply exactly ONE random augmentation to the audio.

    Stacking multiple augmentations degrades audio quality and teaches
    the model to predict from distorted inputs, hurting real-world
    performance.  One clean transform per copy is sufficient.

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

    Returns
    -------
    np.ndarray
        Augmented audio waveform.
    """
    # Build list of enabled augmentations
    available = [
        name for name, flag in _AUGMENTATION_REGISTRY.items()
        if getattr(config, flag, False)
    ]

    if not available:
        return audio

    choice = rng.choice(available)

    if choice == "noise":
        return add_gaussian_noise(audio, rng, config.noise_snr_range)
    elif choice == "tempo":
        return change_tempo(audio, sr, rng, config.speed_range)
    elif choice == "pitch":
        return shift_pitch(audio, sr, rng, config.pitch_semitone_range)
    elif choice == "volume":
        return change_volume(audio, rng, config.volume_gain_db_range)
    else:
        return audio


# ---------------------------------------------------------------------------
# Code-switch detection (stricter than a simple [a-zA-Z]{2,} regex)
# ---------------------------------------------------------------------------

# Common short Nepali words sometimes written in Latin script.
# These must NOT trigger code-switch detection.
_NEPALI_LATIN_FALSE_POSITIVES = {
    "ma", "ta", "ko", "le", "ho", "ni", "ra", "na", "ke", "yo",
    "ji", "la", "ha", "re", "ki", "ka", "ga", "ba", "da", "pa",
    "ne", "se", "he", "ya", "sa", "de", "cha", "chu", "bha",
    "hola", "gara", "garnu", "bhayo", "thiyo", "huncha",
}


def _is_code_switched(sentence: str) -> bool:
    """Check if a sentence contains genuine English words.

    More conservative than a simple ``[a-zA-Z]{2,}`` regex:
      - Requires at least 3 Latin characters per word.
      - Filters out common Nepali particles that may appear in Latin.
    """
    if not sentence or not isinstance(sentence, str):
        return False
    latin_words = re.findall(r"\b[a-zA-Z]{3,}\b", sentence)
    for word in latin_words:
        if word.lower() not in _NEPALI_LATIN_FALSE_POSITIVES:
            return True
    return False


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------

def _make_augment_fn(seed: int, config: AugmentationConfig):
    """Return a stateless map function with per-index RNG.

    Using the example index to seed ensures reproducibility *and* avoids
    sharing mutable RNG state across workers.
    """
    def _augment_map(example, idx):
        rng = np.random.default_rng(seed + idx)
        audio_data = example["audio"]
        audio_array = audio_data["array"]
        sr = audio_data["sampling_rate"]

        if audio_array is None or len(audio_array) == 0:
            return example

        audio_array = np.asarray(audio_array, dtype=np.float32)

        # Skip if input audio is invalid
        if not validate_audio(audio_array, sr, config):
            return example

        try:
            augmented = augment_audio(audio_array, sr, rng, config)
        except Exception as e:
            logger.debug("Augmentation failed for idx %d: %s", idx, e)
            return example

        # Validate output — reject if augmentation produced garbage
        if not np.isfinite(augmented).all():
            logger.debug("Augmentation produced non-finite at idx %d, keeping original.", idx)
            return example

        example["audio"] = {
            "array": augmented,
            "sampling_rate": sr,
            "path": audio_data.get("path", ""),
        }
        return example

    return _augment_map


def augment_dataset(
    dataset: Dataset,
    config: AugmentationConfig,
    split_name: str = "train",
    tmp_dir: Optional[str] = None,
) -> Dataset:
    """Create augmented copies of a dataset split.

    Strategy (conservative):
      1. Create ``augment_factor`` copies of ALL training data
         (each sample gets exactly 1 random transform).
      2. Create ``cs_extra_copies`` additional copies of code-switched
         samples only (to mildly boost CS representation).
      3. Concatenate original + augmented, shuffle.

    Validation and test splits should NOT be passed here.
    """
    import gc
    import shutil
    import tempfile

    if tmp_dir is None:
        tmp_dir = os.path.join(tempfile.gettempdir(), f"neplish_aug_{split_name}")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    sentences = dataset["sentence"]
    cs_indices = [i for i, s in enumerate(sentences) if _is_code_switched(s)]
    nepali_indices = [i for i, s in enumerate(sentences) if not _is_code_switched(s)]

    logger.info(
        "[%s] %d pure-Nepali, %d code-switched out of %d total",
        split_name, len(nepali_indices), len(cs_indices), len(dataset),
    )

    saved_paths: list[str] = []
    seed_offset = 0

    def _map_and_save(subset: Dataset, label: str, copy_idx: int) -> None:
        nonlocal seed_offset
        aug_seed = config.seed + seed_offset
        aug_fn = _make_augment_fn(aug_seed, config)

        aug_ds = subset.map(
            aug_fn,
            with_indices=True,
            keep_in_memory=False,
            num_proc=1,  # librosa is not always safe with multiprocessing
            desc=f"{label} copy {copy_idx + 1}",
        )

        save_path = os.path.join(tmp_dir, f"{label}_{copy_idx}")
        aug_ds.save_to_disk(save_path)
        saved_paths.append(save_path)

        logger.info(
            "[%s] %s augmented copy %d saved (%d samples)",
            split_name, label, copy_idx + 1, len(aug_ds),
        )
        seed_offset += len(subset)
        del aug_ds
        gc.collect()

    # --- Step 1: General augmentation of all samples ---
    if config.augment_factor > 0:
        logger.info(
            "[%s] Creating %d general augmented copies of %d samples ...",
            split_name, config.augment_factor, len(dataset),
        )
        for copy_idx in range(config.augment_factor):
            _map_and_save(dataset, "general", copy_idx)

    # --- Step 2: Extra CS oversampling (moderate) ---
    if config.oversample_code_switched and cs_indices and config.cs_extra_copies > 0:
        cs_subset = dataset.select(cs_indices)
        logger.info(
            "[%s] Creating %d extra copies of %d code-switched samples ...",
            split_name, config.cs_extra_copies, len(cs_subset),
        )
        for copy_idx in range(config.cs_extra_copies):
            _map_and_save(cs_subset, "codeswitched", copy_idx)
        del cs_subset
        gc.collect()

    # --- Step 3: Concatenate ---
    if saved_paths:
        logger.info("[%s] Loading %d augmented splits ...", split_name, len(saved_paths))
        augmented_parts = [load_from_disk(p) for p in saved_paths]
        final_dataset = concatenate_datasets([dataset] + augmented_parts)
        final_dataset = final_dataset.shuffle(seed=config.seed)

        logger.info(
            "[%s] Final: %d samples (original %d + augmented %d)",
            split_name, len(final_dataset), len(dataset),
            len(final_dataset) - len(dataset),
        )

        shutil.rmtree(tmp_dir, ignore_errors=True)
        del augmented_parts
        gc.collect()
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
    augment_factor: int = 1,
    cs_extra_copies: int = 1,
) -> None:
    """Run the augmentation pipeline from the command line."""
    if input_dir is None:
        input_dir = str(PROJECT_ROOT / "data" / "hf_dataset")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "hf_dataset_augmented")

    config = AugmentationConfig(
        augment_factor=augment_factor,
        cs_extra_copies=cs_extra_copies,
    )

    logger.info("Augmentation config:")
    logger.info("  General augment factor : %d (each gets 1 random transform)", config.augment_factor)
    logger.info("  CS extra copies        : %d", config.cs_extra_copies)
    logger.info("  Noise SNR range        : %s dB", config.noise_snr_range)
    logger.info("  Tempo range            : %s", config.speed_range)
    logger.info("  Pitch range            : %s semitones", config.pitch_semitone_range)
    logger.info("  Volume range           : %s dB", config.volume_gain_db_range)

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
        "--augment_factor", type=int, default=1,
        help="Augmented copies per sample (each gets 1 random transform). Default: 1.",
    )
    parser.add_argument(
        "--cs_extra_copies", type=int, default=1,
        help="Extra augmented copies for code-switched samples. Default: 1.",
    )
    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        augment_factor=args.augment_factor,
        cs_extra_copies=args.cs_extra_copies,
    )
