# Skill: Model Training

## Description
Fine-tune Whisper-large-v3 with QLoRA for Neplish code-switched ASR.

## Prerequisites
1. Dataset prepared and augmented (see `data-preparation.md`)
2. Dependencies installed: `pip install -r requirements.txt`
3. NVIDIA GPU with CUDA support (configured for RTX 4090)

## Run Training
```bash
python -m src.training.train
```

## Configuration
All settings are in `src/training/config.py`:

| Parameter | Value | Notes |
|---|---|---|
| Model | `openai/whisper-large-v3` | 1.55B params |
| Quantization | 4-bit NF4 | ~2 GB for weights |
| LoRA rank | 32 | Higher rank = more capacity |
| LoRA alpha | 64 | Scaling factor |
| Batch size | 16 | Per device (RTX 4090) |
| Grad accumulation | 2 | Effective batch = 32 |
| Precision | bfloat16 | Native Ada Lovelace support |
| Epochs | 8 | |
| Learning rate | 1e-4 | Typical for QLoRA |
| Optimizer | paged_adamw_8bit | Memory efficient |
| Dataset | augmented by default | `use_augmented=True` |

## Switching to Original Dataset
In `src/training/config.py`, set:
```python
use_augmented: bool = False
```

## Output
- Checkpoints: `models/whisper-neplish/checkpoint-*/`
- Final model: `models/whisper-neplish/final/`
- Best model selected by lowest validation WER

## Memory Estimates (RTX 4090 24 GB)
- Base model (4-bit): ~2 GB
- LoRA adapters (bf16): ~0.5 GB
- Optimizer states (8-bit): ~1 GB
- Activations (batch=16 + grad ckpt): ~10-14 GB
- Total: ~14-18 GB (comfortable headroom)
