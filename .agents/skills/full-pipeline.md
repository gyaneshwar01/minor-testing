# Skill: Full Pipeline (End-to-End)

## Description
Complete workflow from raw data to trained model and evaluation.

## Step-by-Step

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Clean Transcripts (skip if `cleaned_dataset.csv` exists)
```bash
python -m src.data.clean_transcripts
```

### 3. Create HuggingFace Dataset (skip if `data/hf_dataset/` exists)
```bash
python -m src.data.prepare_dataset
```

### 4. Augment Dataset
```bash
python -m src.data.augment_dataset
```

### 5. Train Model
```bash
python -m src.training.train
```

### 6. Evaluate
```bash
# Fine-tuned model
python -m src.evaluation.evaluate --model_dir models/whisper-neplish/final

# Baseline comparison
python -m src.evaluation.run_baseline

# Side-by-side analysis
python -m src.evaluation.analyze_results \
    --finetuned_results evaluation_results.json \
    --baseline_results evaluation_results_baseline.json
```

### 7. Demo
```bash
python -m src.demo.app --share
```

## Troubleshooting

### CUDA OOM during training
- Reduce `per_device_train_batch_size` in `src/training/config.py`
- Increase `gradient_accumulation_steps` to maintain effective batch size

### Augmentation OOM (RAM)
- Reduce `augment_factor`, `nepali_factor`, or `english_factor`
- Process in smaller chunks

### Dataset not found errors
- Ensure you ran steps 2-4 before training
- Check paths in `src/training/config.py` match your directory structure
- If augmented dataset missing, training falls back to original automatically
