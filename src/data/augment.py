"""
Audio Data Augmentation Pipeline for Neplish ASR.

Applies audio-domain augmentations that do NOT alter the transcript labels.
All transforms operate on the waveform only, so the ground-truth text stays
the same.  This effectively multiplies the training set size.

Supported augmentations:
  - Speed perturbation (0.9x – 1.1x)
  - Pitch shifting (±2 semitones)
  - Adding background Gaussian noise
  - Time masking (SpecAugment-style, on the waveform)
  - Volume perturbation (gain ±6 dB)

Usage:
    python -m src.data.augment [--dataset_dir data/hf_dataset]
                                [--output_dir data/hf_dataset_augmented]
                                [--num_augmented_copies 2]
"""

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Individual augmentation functions (pure numpy, no extra deps required)
# ---------------------------------------------------------------------------

def speed_perturbation(
    audio: np.ndarray,
    sr: int,
    factor: float | None = None,
    min_factor: float = 0.9,
    max_factor: float = 1.1,
) -> np.ndarray:
    """Change playback speed without pitch correction.

    Resamples the audio to simulate speaking faster/slower.  Since this
    changes duration but NOT the transcript content, labels remain valid.
    """
    if factor is None:
        factor = random.uniform(min_factor, max_factor)

    # Resample using linear interpolation (fast, dependency-free)
    indices = np.arange(0, len(audio), factor)
    indices = indices[indices < len(audio)].astype(int)
    return audio[indices].astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float | None = None,
    max_semitones: float = 2.0,
) -> np.ndarray:
    """Shift pitch by resampling trick (speed change + resample back).

    This approximation avoids heavy DSP deps.  For training augmentation
    the slight artefacts actually help generalisation.
    """
    if semitones is None:
        semitones = random.uniform(-max_semitones, max_semitones)

    # Pitch shift via speed change ratio
    ratio = 2.0 ** (semitones / 12.0)
    # Change speed (changes pitch)
    indices = np.arange(0, len(audio), ratio)
    indices = indices[indices < len(audio)].astype(int)
    shifted = audio[indices]

    # Resample back to original length to keep duration roughly the same
    target_len = len(audio)
    if len(shifted) == 0:
        return audio.copy()
    x_old = np.linspace(0, 1, len(shifted))
    x_new = np.linspace(0, 1, target_len)
    resampled = np.interp(x_new, x_old, shifted)
    return resampled.astype(np.float32)


def add_gaussian_noise(
    audio: np.ndarray,
    sr: int,
    snr_db: float | None = None,
    min_snr_db: float = 15.0,
    max_snr_db: float = 35.0,
) -> np.ndarray:
    """Add Gaussian white noise at a random SNR."""
    if snr_db is None:
        snr_db = random.uniform(min_snr_db, max_snr_db)

    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio.copy()

    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)


def volume_perturbation(
    audio: np.ndarray,
    sr: int,
    gain_db: float | None = None,
    min_gain_db: float = -6.0,
    max_gain_db: float = 6.0,
) -> np.ndarray:
    """Apply random gain (volume change) in dB."""
    if gain_db is None:
        gain_db = random.uniform(min_gain_db, max_gain_db)

    gain_linear = 10 ** (gain_db / 20)
    augmented = audio * gain_linear

    # Clip to prevent clipping distortion
    augmented = np.clip(augmented, -1.0, 1.0)
    return augmented.astype(np.float32)


def time_mask(
    audio: np.ndarray,
    sr: int,
    max_mask_fraction: float = 0.1,
    num_masks: int = 2,
) -> np.ndarray:
    """Zero out random contiguous segments (SpecAugment-style on waveform).

    The masked duration is short enough that the overall transcript stays
    intelligible, acting as a regulariser.
    """
    augmented = audio.copy()
    max_mask_len = int(len(audio) * max_mask_fraction)

    for _ in range(num_masks):
        mask_len = random.randint(1, max(1, max_mask_len))
        start = random.randint(0, max(0, len(audio) - mask_len))
        augmented[start : start + mask_len] = 0.0

    return augmented.astype(np.float32)


# ---------------------------------------------------------------------------
# Composite augmentation pipeline
# ---------------------------------------------------------------------------

# Registry of available transforms (all are label-preserving)
AUGMENTATIONS = {
    "speed": speed_perturbation,
    "pitch": pitch_shift,
    "noise": add_gaussian_noise,
    "volume": volume_perturbation,
    "time_mask": time_mask,
}


def random_augment(
    audio: np.ndarray,
    sr: int,
    min_transforms: int = 1,
    max_transforms: int = 3,
    enabled_transforms: list[str] | None = None,
) -> np.ndarray:
    """Apply a random subset of augmentations to an audio sample.

    Parameters
    ----------
    audio : np.ndarray
        Audio waveform (float32, mono).
    sr : int
        Sampling rate (expected 16000 for Whisper).
    min_transforms / max_transforms : int
        How many augmentations to chain.
    enabled_transforms : list[str] or None
        Which transforms to sample from.  Defaults to all.

    Returns
    -------
    np.ndarray
        Augmented audio waveform (same sampling rate, labels unchanged).
    """
    if enabled_transforms is None:
        enabled_transforms = list(AUGMENTATIONS.keys())

    num = random.randint(min_transforms, min(max_transforms, len(enabled_transforms)))
    chosen = random.sample(enabled_transforms, num)

    augmented = audio.copy()
    for name in chosen:
        fn = AUGMENTATIONS[name]
        augmented = fn(augmented, sr)

    return augmented


# ---------------------------------------------------------------------------
# Dataset-level augmentation
# ---------------------------------------------------------------------------

def augment_dataset(
    dataset: DatasetDict,
    num_augmented_copies: int = 2,
    augment_train_only: bool = True,
    seed: int = 42,
    max_audio_length_s: float = 30.0,
) -> DatasetDict:
    """Create augmented copies of the training split and concatenate.

    The original (clean) training data is always kept.  We add
    ``num_augmented_copies`` augmented versions on top.

    Validation and test splits are left unchanged so that evaluation
    is always on clean data.

    Parameters
    ----------
    dataset : DatasetDict
        Original dataset with 'train', 'validation', 'test' splits.
    num_augmented_copies : int
        How many augmented copies of each training sample to create.
    augment_train_only : bool
        If True, only augment the 'train' split.
    seed : int
        Random seed for reproducibility.
    max_audio_length_s : float
        Maximum audio length in seconds.  Augmented clips exceeding this
        are skipped (Whisper cannot handle >30s).

    Returns
    -------
    DatasetDict
        Dataset with augmented training split.
    """
    random.seed(seed)
    np.random.seed(seed)

    result = {}

    for split_name, split_ds in dataset.items():
        if augment_train_only and split_name != "train":
            result[split_name] = split_ds
            logger.info("  %s: kept unchanged (%d samples)", split_name, len(split_ds))
            continue

        logger.info(
            "  Augmenting '%s' split (%d originals × %d copies) ...",
            split_name, len(split_ds), num_augmented_copies,
        )

        augmented_audios = []
        augmented_sentences = []

        for copy_idx in range(num_augmented_copies):
            for i in range(len(split_ds)):
                sample = split_ds[i]
                audio_array = sample["audio"]["array"]
                sr = sample["audio"]["sampling_rate"]
                sentence = sample["sentence"]

                # Apply random augmentation
                aug_audio = random_augment(audio_array, sr)

                # Check duration limit (Whisper max = 30s)
                duration_s = len(aug_audio) / sr
                if duration_s > max_audio_length_s:
                    # Truncate rather than skip to keep training set balanced
                    max_samples = int(max_audio_length_s * sr)
                    aug_audio = aug_audio[:max_samples]

                augmented_audios.append({
                    "array": aug_audio,
                    "sampling_rate": sr,
                    "path": None,
                })
                augmented_sentences.append(sentence)

            logger.info("    Copy %d/%d done.", copy_idx + 1, num_augmented_copies)

        # Build augmented dataset
        if augmented_audios:
            aug_ds = Dataset.from_dict({
                "audio": augmented_audios,
                "sentence": augmented_sentences,
            })
            # Cast audio column so HF knows it's audio
            aug_ds = aug_ds.cast_column("audio", Audio(sampling_rate=16_000))

            # Concatenate original + augmented
            combined = concatenate_datasets([split_ds, aug_ds])
            # Shuffle so augmented samples are interspersed
            combined = combined.shuffle(seed=seed)
            result[split_name] = combined

            logger.info(
                "  %s: %d original + %d augmented = %d total",
                split_name, len(split_ds), len(augmented_audios), len(combined),
            )
        else:
            result[split_name] = split_ds

    return DatasetDict(result)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(
    dataset_dir: str | None = None,
    output_dir: str | None = None,
    num_augmented_copies: int = 2,
    seed: int = 42,
) -> None:
    """Run the augmentation pipeline from the command line."""
    if dataset_dir is None:
        dataset_dir = str(PROJECT_ROOT / "data" / "hf_dataset")
    if output_dir is None:
        output_dir = str(PROJECT_ROOT / "data" / "hf_dataset_augmented")

    if not Path(dataset_dir).exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            "Run `python -m src.data.prepare_dataset` first."
        )

    logger.info("Loading dataset from %s", dataset_dir)
    dataset = load_from_disk(dataset_dir)
    logger.info("Original sizes: %s", {k: len(v) for k, v in dataset.items()})

    logger.info("Augmenting with %d copies per training sample ...", num_augmented_copies)
    augmented = augment_dataset(
        dataset,
        num_augmented_copies=num_augmented_copies,
        seed=seed,
    )

    logger.info("Saving augmented dataset to %s ...", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    augmented.save_to_disk(output_dir)

    print(f"\nAugmented dataset saved to {output_dir}")
    print(augmented)
    for split_name, split_ds in augmented.items():
        print(f"  {split_name}: {len(split_ds)} samples")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Augment Neplish ASR training data with audio-domain transforms."
    )
    parser.add_argument(
        "--dataset_dir", type=str, default=None,
        help="Path to the original HF dataset directory.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to save the augmented HF dataset.",
    )
    parser.add_argument(
        "--num_copies", type=int, default=2,
        help="Number of augmented copies per training sample (default: 2).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_augmented_copies=args.num_copies,
        seed=args.seed,
    )
