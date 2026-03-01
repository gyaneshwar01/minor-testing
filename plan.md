# Neplish ASR -- Project Plan & Progress

## Overview

Fine-tune **OpenAI Whisper Large-v3** (1.5B params) for **code-switched Nepali-English
(Neplish)** automatic speech recognition using **QLoRA** (4-bit NF4 quantization +
LoRA). Runs on an **RTX 4090 24 GB** college server GPU.

The dataset contains ~6,800 annotated utterances (~20 hours) across 6 batches, with
~31% of samples containing English code-switching.  Training data is further expanded
with audio augmentation and external pure-Nepali/English audio from FLEURS.

---

## Dataset Summary

| Metric                     | Value                                     |
|----------------------------|-------------------------------------------|
| Total samples              | 6,800                                     |
| Batches                    | 6 (Batch_33 through Batch_38_Modify)      |
| Audio format               | MP3, 16 kHz mono                          |
| Avg duration               | ~10.7 s                                   |
| Estimated total audio      | ~20 hours                                 |
| Code-switched samples      | 2,145 (31.5%)                             |
| Pure Nepali samples        | 4,655 (68.5%)                             |
| Unique English CS words    | ~180                                      |
| Missing files / null vals  | 0                                         |

### Annotation Format

The raw transcripts use bracket notation:

- `[English/Nepali]` -- code-switch marker, e.g. `[digital/डिजिटल]`
- `[word/<unk>]` -- uncertain transcription (9,170 occurrences in all 6,800 samples)
- `[<unk>]` -- completely unintelligible (915 samples)
- `[<sil>]` -- silence (1,048 samples)
- `[<rep>/word]` -- repetition/disfluency (849 samples)
- `[<fin>]` -- end of utterance (657 samples)
- `[<fil>/word]` -- filler word (371 samples)

### Cleaning Strategy

| Pattern            | Resolution                                      |
|--------------------|--------------------------------------------------|
| `[English/Nepali]` | Keep English word (what was actually spoken)      |
| `[word/<unk>]`     | Keep the attempted word (trust transcriber)       |
| `[<unk>]`          | Remove (unintelligible)                           |
| `[<sil>]`          | Remove (silence not transcribed)                  |
| `[<rep>/word]`     | Keep the word (it was spoken)                     |
| `[<fin>]`          | Remove (metadata)                                 |
| `[<fil>/word]`     | Keep the word (filler was spoken)                 |

---

## Project Structure

```
minor-project/
├── Dataset/                           # Raw data (6,800 audio + annotations)
│   ├── combined_dataset.csv           # Raw annotated transcripts
│   └── cleaned_dataset.csv            # Cleaned transcripts (generated)
├── data/
│   ├── hf_dataset/                    # HuggingFace DatasetDict (generated)
│   │   ├── train/                     # 5,440 samples (80%)
│   │   ├── validation/                # 680 samples (10%)
│   │   └── test/                      # 680 samples (10%)
│   ├── hf_dataset_augmented/          # Augmented training set (generated)
│   │   ├── train/                     # ~16,000+ samples (original + augmented + external)
│   │   ├── validation/                # 680 samples (unchanged)
│   │   └── test/                      # 680 samples (unchanged)
│   └── external/                      # External pure-language data (generated)
│       ├── nepali/                    # FLEURS Nepali (~500 samples)
│       └── english/                   # FLEURS English (~300 samples)
├── models/
│   └── whisper-neplish/               # Training checkpoints + final model
│       └── final/                     # Best model (generated after training)
├── src/
│   ├── data/
│   │   ├── clean_transcripts.py       # Bracket notation parser
│   │   ├── prepare_dataset.py         # CSV -> HF Dataset + splits + augment
│   │   ├── augment.py                 # Audio augmentation pipeline
│   │   └── add_external_audio.py      # Download FLEURS Nepali + English
│   ├── training/
│   │   ├── config.py                  # All hyperparameters (QLoRA + training)
│   │   └── train.py                   # Whisper + QLoRA fine-tuning
│   ├── evaluation/
│   │   ├── evaluate.py                # WER, CER, CS-WER, English Token F1
│   │   ├── run_baseline.py            # Vanilla Whisper baseline runner
│   │   └── analyze_results.py         # Side-by-side comparison + charts
│   └── demo/
│       └── app.py                     # Gradio UI (mic, upload, comparison)
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   └── 02_training.ipynb              # GPU-ready QLoRA training notebook
├── requirements.txt                   # All dependencies (incl. bitsandbytes)
└── plan.md                            # This file
```

---

## Phases & Progress

### Phase 1: Data Preparation -- DONE ✅

| Task                       | Status | File                          | Notes                                     |
|----------------------------|--------|-------------------------------|-------------------------------------------|
| Transcript cleaning        | DONE   | `src/data/clean_transcripts.py` | Tested; produces `cleaned_dataset.csv`  |
| HF Dataset + splits        | DONE   | `src/data/prepare_dataset.py`   | Tested; 5440/680/680 stratified split   |
| EDA notebook               | DONE   | `notebooks/01_eda.ipynb`        | 9 sections with visualisations          |

### Phase 1.5: Data Augmentation & External Audio -- NEW 🆕

| Task                       | Status | File                               | Notes                                     |
|----------------------------|--------|-------------------------------------|-------------------------------------------|
| Audio augmentation         | DONE   | `src/data/augment.py`               | Speed, pitch, noise, volume, time-mask  |
| External audio (FLEURS)    | DONE   | `src/data/add_external_audio.py`    | Pure Nepali + English from FLEURS       |
| Integrated pipeline        | DONE   | `src/data/prepare_dataset.py`       | `--augment --add_external` flags        |

**Augmentation transforms (all label-preserving):**

| Transform           | Range                      | Effect                              |
|---------------------|----------------------------|--------------------------------------|
| Speed perturbation  | 0.9x – 1.1x               | Simulates fast/slow speakers         |
| Pitch shift         | ±2 semitones               | Simulates different vocal ranges     |
| Gaussian noise      | SNR 15-35 dB               | Robustness to noisy environments     |
| Volume perturbation | ±6 dB                      | Microphone distance variation        |
| Time masking        | Up to 10% per mask, ×2     | SpecAugment-style regularisation     |

**External data sources:**

| Source            | Language | Samples | Purpose                              |
|-------------------|----------|---------|--------------------------------------|
| FLEURS (ne_np)    | Nepali   | ~500    | Improve pure-Nepali recognition      |
| FLEURS (en_us)    | English  | ~300    | Improve English word recognition     |

**How to run:**

```bash
# Step 1: Clean transcripts (if not done)
python -m src.data.clean_transcripts

# Step 2: Create base dataset + augmented dataset + external audio
python -m src.data.prepare_dataset --augment --add_external

# Or run augmentation separately:
python -m src.data.augment --num_copies 2

# Or download external data separately:
python -m src.data.add_external_audio --max_nepali 500 --max_english 300
```

### Phase 2: Model Training -- READY TO RUN 🚀

| Task                       | Status       | File                          | Notes                                  |
|----------------------------|--------------|-------------------------------|----------------------------------------|
| Training config (QLoRA)    | DONE         | `src/training/config.py`      | 4-bit NF4 + LoRA r=32 + bf16          |
| Training script            | DONE         | `src/training/train.py`       | Whisper Large-v3 + QLoRA via PEFT      |
| Training notebook          | UPDATE       | `notebooks/02_training.ipynb` | Needs update for large-v3              |

**Model:** `openai/whisper-large-v3` (1.5B params) with QLoRA:
- Base model quantized to 4-bit NF4 (~3 GB VRAM)
- LoRA adapters (r=32, alpha=64) trained in bf16
- Target modules: q/v/k/o projections + fc1/fc2
- Total VRAM usage: ~12-16 GB (fits comfortably on RTX 4090 24 GB)

**To run on the server:**

```bash
python -m src.training.train
```

**Key training hyperparameters:**

| Parameter                  | Value              |
|----------------------------|--------------------|  
| Base model                 | whisper-large-v3   |
| Quantization               | 4-bit NF4 (QLoRA) |
| Double quantization        | Yes (~0.4 GB saved)|
| LoRA rank (r)              | 32                 |
| LoRA alpha                 | 64                 |
| LoRA targets               | q/v/k/o + fc1/fc2  |
| Learning rate              | 5e-5               |
| Optimizer                  | paged_adamw_8bit   |
| Epochs                     | 10                 |
| Batch size                 | 16                 |
| Gradient accumulation      | 2                  |
| Effective batch size       | 32                 |
| Warmup steps               | 500                |
| Eval/save every            | 200 steps          |
| Precision                  | bf16               |
| Gradient checkpointing     | No (24 GB enough)  |

**Why Whisper Large-v3 over Medium?**
- Large-v3 (1.5B params) vs Medium (769M): ~2x capacity
- Better multilingual & code-switching performance out of the box
- RTX 4090 24 GB has plenty of VRAM for 4-bit large-v3 + LoRA
- bf16 native on Ada Lovelace = faster training than fp16

**Why QLoRA still?**
- Large-v3 in fp16 = ~3 GB; in 4-bit = ~1.5 GB for weights
- Remaining VRAM for activations, optimizer states, large batch sizes
- Same quality as full fine-tuning per the QLoRA paper
- Allows batch size 16 with no gradient checkpointing = fast training

### Phase 3: Evaluation -- DONE ✅

| Task                       | Status       | File                              | Notes                              |
|----------------------------|--------------|-----------------------------------|------------------------------------|
| Evaluation pipeline        | DONE         | `src/evaluation/evaluate.py`      | WER, CER, CS-WER, Eng Token F1    |
| Baseline runner            | DONE         | `src/evaluation/run_baseline.py`  | Vanilla whisper-large-v3 on test set |
| Results comparison         | DONE         | `src/evaluation/analyze_results.py`| Charts + tables                   |

**To run (after training is complete):**

```bash
# Evaluate fine-tuned model
python -m src.evaluation.evaluate --model_dir models/whisper-neplish/final

# Run vanilla Whisper baseline
python -m src.evaluation.run_baseline

# Compare results
python -m src.evaluation.analyze_results \
    --finetuned_results evaluation_results.json \
    --baseline_results evaluation_results_baseline.json
```

**Metrics computed:**

1. **WER** -- Word Error Rate (primary metric)
2. **CER** -- Character Error Rate (important for Devanagari)
3. **Code-Switch WER** -- WER on the ~31% code-switched subset
4. **Nepali-only WER** -- WER on pure-Nepali subset
5. **English Token Precision / Recall / F1** -- accuracy on English words

### Phase 4: Demo -- RUNNING ✅

| Task                       | Status       | File                    | Notes                                 |
|----------------------------|--------------|-------------------------|---------------------------------------|
| Gradio app                 | DONE         | `src/demo/app.py`       | Mic + upload + model comparison       |

**To run (after training):**

```bash
python -m src.demo.app          # local at localhost:7860
python -m src.demo.app --share  # public Gradio link (for demo day)
```

**Features:**
- Tab 1: Transcribe with fine-tuned model (mic or file upload)
- Tab 2: Side-by-side comparison (vanilla Whisper vs fine-tuned)
- Tab 3: About page with technical details
- English words highlighted in bold in output

---

## Technology Stack

| Component         | Library                                  |
|-------------------|------------------------------------------|
| Base model        | `openai/whisper-large-v3` via transformers|
| Quantization      | 4-bit NF4 via `bitsandbytes`             |
| Fine-tuning       | QLoRA (LoRA + 4-bit) via `peft`          |
| Optimizer         | paged AdamW 8-bit via `bitsandbytes`     |
| Training          | `Seq2SeqTrainer` from transformers       |
| Dataset handling  | HuggingFace `datasets`                   |
| Audio processing  | `librosa`, `soundfile`                   |
| Augmentation      | Custom numpy-based (no extra deps)       |
| External data     | Google FLEURS via HuggingFace Hub        |
| Metrics           | `jiwer`, `evaluate`                      |
| Demo UI           | `gradio`                                 |
| Experiment track  | `wandb` (optional, off by default)       |

### Hardware

| Component         | Spec                                     |
|-------------------|------------------------------------------|
| GPU               | NVIDIA GeForce RTX 4090                  |
| VRAM              | 24 GB                                    |
| Architecture      | Ada Lovelace (native bf16)               |

---

## Remaining Work (in order)

1. **Install full dependencies** -- `pip install -r requirements.txt`
2. **Prepare augmented dataset** -- `python -m src.data.prepare_dataset --augment --add_external`
3. **Run training** -- `python -m src.training.train` (uses augmented dataset by default)
4. **Run evaluation** -- baseline + fine-tuned comparison
5. **Launch demo** -- Gradio app for college showcase
6. **Optional enhancements:**
   - Push model to HuggingFace Hub
   - Add W&B experiment tracking (`report_to="wandb"` in config)
   - Try even larger LoRA rank (r=64) if metrics plateau
   - Experiment with more augmentation copies (--num_copies 3)
