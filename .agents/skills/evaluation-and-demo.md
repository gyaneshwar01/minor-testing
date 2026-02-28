# Skill: Evaluation & Demo

## Description
Evaluate fine-tuned models and run the Gradio demo app.

## Evaluation

### Evaluate Fine-tuned Model
```bash
python -m src.evaluation.evaluate --model_dir models/whisper-neplish/final
```
- Computes: WER, CER, Code-Switch WER, English Token F1
- Output: `evaluation_results.json`

### Run Baseline (Vanilla Whisper)
```bash
python -m src.evaluation.run_baseline
```
- Uses vanilla `whisper-large-v3` without fine-tuning
- Output: `evaluation_results_baseline.json`

### Compare Results
```bash
python -m src.evaluation.analyze_results \
    --finetuned_results evaluation_results.json \
    --baseline_results evaluation_results_baseline.json
```
- Generates comparison charts and tables

## Demo App

### Local
```bash
python -m src.demo.app
```
- Runs at `http://localhost:7860`

### Public Link (for demo day)
```bash
python -m src.demo.app --share
```
- Creates a public Gradio share link

### Features
- Tab 1: Transcribe with fine-tuned model (mic or file upload)
- Tab 2: Side-by-side vanilla vs fine-tuned comparison
- Tab 3: About page with technical details
- English words highlighted in bold in output

## Key Metrics
| Metric | What it measures |
|---|---|
| WER | Overall word error rate |
| CER | Character error rate (important for Devanagari) |
| CS-WER | WER on code-switched subset (~31% of data) |
| Nepali-WER | WER on pure-Nepali subset |
| Eng Token F1 | Precision/Recall/F1 for English word detection |
