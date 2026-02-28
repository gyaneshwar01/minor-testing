# Project Rules for Neplish ASR

## Architecture
- Follow the existing `src/` module structure: `data/`, `training/`, `evaluation/`, `demo/`
- All scripts must be runnable as modules: `python -m src.<subpackage>.<module>`
- Configuration is centralised in `src/training/config.py` — never hardcode hyperparameters elsewhere
- Use dataclasses for config; avoid loose globals or magic numbers

## Model
- Base model: `openai/whisper-large-v3` (1.55B params)
- Fine-tuning method: QLoRA (4-bit NF4 quantization + LoRA via PEFT)
- Target hardware: RTX 4090 (24 GB VRAM), Intel i9 14th Gen, 64 GB RAM
- Always use `bfloat16` compute dtype on Ada Lovelace GPUs
- LoRA targets: `q_proj`, `v_proj`, `k_proj`, `o_proj` in attention layers

## Data
- Dataset: ~6,800 Nepali-English code-switched utterances (~20 hours)
- Audio: 16 kHz mono, max 30 seconds (Whisper limit)
- Transcripts use bracket annotation format — see `plan.md` for full spec
- Cleaning is handled by `src/data/clean_transcripts.py`
- Augmentation is handled by `src/data/augment_dataset.py`
- Never modify test/validation splits during augmentation — only augment train

## Code Style
- Python 3.10+ (use `X | Y` union types, not `Union[X, Y]`)
- Use type hints on all function signatures
- Use `logging` module, not `print()`, for status messages (except final summaries)
- Docstrings: NumPy style
- Imports: stdlib → third-party → local, separated by blank lines

## Git
- Large files (audio, .arrow, model weights) must NEVER be committed
- `.gitignore` covers: `Dataset/`, `data/`, `hf_dataset*/`, `models/` (except final adapter)
- Use forward slashes in `.gitignore` patterns

## Dependencies
- All packages listed in `requirements.txt`
- Key libraries: `transformers`, `peft`, `bitsandbytes`, `datasets`, `librosa`, `jiwer`, `gradio`
- `scipy` is required for augmentation (pink noise filter)
