# Neplish ASR -- Project Plan & Progress

## Overview

Fine-tune **OpenAI Whisper Medium** (769M params) for **code-switched Nepali-English
(Neplish)** automatic speech recognition using **QLoRA** (4-bit NF4 quantization +
LoRA). This fits the full medium model on a 6 GB GPU (RTX 4050 Laptop).

The dataset contains ~6,800 annotated utterances (~20 hours) across 6 batches, with
~31% of samples containing English code-switching.

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
│   └── hf_dataset/                    # HuggingFace DatasetDict (generated)
│       ├── train/                     # 5,440 samples (80%)
│       ├── validation/                # 680 samples (10%)
│       └── test/                      # 680 samples (10%)
├── models/
│   └── whisper-neplish/               # Training checkpoints + final model
│       └── final/                     # Best model (generated after training)
├── src/
│   ├── data/
│   │   ├── clean_transcripts.py       # Bracket notation parser
│   │   └── prepare_dataset.py         # CSV -> HF Dataset + stratified splits
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

**How to run (already done once):**

```bash
# Step 1: Clean transcripts
python -m src.data.clean_transcripts

# Step 2: Create HF dataset with train/val/test splits
python -m src.data.prepare_dataset
```

### Phase 2: Model Training -- DONE ✅

| Task                       | Status       | File                          | Notes                                  |
|----------------------------|--------------|-------------------------------|----------------------------------------|
| Training config (QLoRA)    | DONE         | `src/training/config.py`      | 4-bit NF4 + LoRA + 8-bit optimizer     |
| Training script            | DONE         | `src/training/train.py`       | Whisper Medium + QLoRA via PEFT        |
| Training notebook          | DONE         | `notebooks/02_training.ipynb` | Completed; test WER = 24.51%           |

**Model:** `openai/whisper-medium` (769M params) with QLoRA:
- Base model quantized to 4-bit NF4 (~1.5 GB VRAM)
- LoRA adapters (r=16, alpha=32) trained in fp16
- Total VRAM usage: ~4.5-5.5 GB (fits in RTX 4050 6 GB)

**To run locally:**

```bash
python -m src.training.train
```

**To run on Colab:** upload the project and run `notebooks/02_training.ipynb`.

**Key training hyperparameters:**

| Parameter                  | Value              |
|----------------------------|--------------------|
| Base model                 | whisper-medium     |
| Quantization               | 4-bit NF4 (QLoRA) |
| Double quantization        | Yes (~0.4 GB saved)|
| LoRA rank (r)              | 16                 |
| LoRA alpha                 | 32                 |
| LoRA targets               | q/v/k/o projections|
| Learning rate              | 1e-4               |
| Optimizer                  | paged_adamw_8bit   |
| Epochs                     | 8                  |
| Batch size                 | 4                  |
| Gradient accumulation      | 4                  |
| Effective batch size       | 16                 |
| Warmup steps               | 200                |
| Eval/save every            | 200 steps          |
| Precision                  | fp16               |
| Gradient checkpointing     | Yes                |

**Why QLoRA over LoRA?**
- Whisper-medium in fp16 needs ~1.5 GB; in 4-bit only ~0.5 GB for weights
- LoRA alone couldn't fit whisper-medium on 6 GB with reasonable batch sizes
- QLoRA: same quality as full fine-tuning per the QLoRA paper, ~60% less memory
- 8-bit paged AdamW further reduces optimizer state memory

### Phase 3: Evaluation -- DONE ✅

| Task                       | Status       | File                              | Notes                              |
|----------------------------|--------------|-----------------------------------|------------------------------------|
| Evaluation pipeline        | DONE         | `src/evaluation/evaluate.py`      | WER, CER, CS-WER, Eng Token F1    |
| Baseline runner            | DONE         | `src/evaluation/run_baseline.py`  | Vanilla whisper-medium on test set |
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
| Base model        | `openai/whisper-medium` via transformers  |
| Quantization      | 4-bit NF4 via `bitsandbytes`             |
| Fine-tuning       | QLoRA (LoRA + 4-bit) via `peft`          |
| Optimizer         | paged AdamW 8-bit via `bitsandbytes`     |
| Training          | `Seq2SeqTrainer` from transformers       |
| Dataset handling  | HuggingFace `datasets`                   |
| Audio processing  | `librosa`, `soundfile`                   |
| Metrics           | `jiwer`, `evaluate`                      |
| Demo UI           | `gradio`                                 |
| Experiment track  | `wandb` (optional, off by default)       |

### Hardware

| Component         | Spec                                     |
|-------------------|------------------------------------------|
| GPU               | NVIDIA GeForce RTX 4050 Laptop GPU       |
| VRAM              | 6 GB (6,141 MiB)                         |
| CUDA              | 13.0                                     |
| Driver            | 580.126.09                               |

---

## Remaining Work (in order)

1. **Install full dependencies** -- `pip install -r requirements.txt`
2. **Run training** -- locally using `python -m src.training.train` or `02_training.ipynb`
3. **Run evaluation** -- baseline + fine-tuned comparison
4. **Launch demo** -- Gradio app for college showcase
5. **Optional enhancements:**
   - Add data augmentation (speed perturbation, noise injection)
   - Push model to HuggingFace Hub
   - Add W&B experiment tracking (`report_to="wandb"` in config)
   - Try larger LoRA rank (r=32) if metrics plateau
