# Skill: Data Preparation Pipeline

## Description
End-to-end data preparation for Neplish ASR training.

## Steps

### 1. Clean Transcripts
```bash
python -m src.data.clean_transcripts
```
- Input: `Dataset/combined_dataset.csv` (raw bracket-annotated transcripts)
- Output: `Dataset/cleaned_dataset.csv`
- Resolves all bracket annotations `[English/Nepali]`, `[<unk>]`, `[<sil>]`, etc.

### 2. Create HuggingFace Dataset
```bash
python -m src.data.prepare_dataset
```
- Input: `Dataset/cleaned_dataset.csv`
- Output: `data/hf_dataset/` (DatasetDict with train/validation/test splits)
- Stratified 80/10/10 split balanced by batch and code-switch presence

### 3. Augment Dataset
```bash
python -m src.data.augment_dataset
```
- Input: `data/hf_dataset/`
- Output: `data/hf_dataset_augmented/`
- Augmentations: white noise, pink noise, speed perturbation, pitch shift, volume change, time shift
- Oversamples code-switched samples (2x) and pure-Nepali samples (1x)
- Only augments training split; validation/test remain untouched

### CLI Options (augmentation)
```bash
python -m src.data.augment_dataset \
    --input_dir data/hf_dataset \
    --output_dir data/hf_dataset_augmented \
    --augment_factor 2 \
    --nepali_factor 1 \
    --english_factor 2
```

## Expected Dataset Sizes
- Original: 5,440 train / 680 val / 680 test
- After augmentation (default): ~27,200 train / 680 val / 680 test
