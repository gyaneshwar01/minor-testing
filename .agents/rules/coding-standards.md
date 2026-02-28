# Coding Standards

## Error Handling
- Wrap audio processing loops in try/except to prevent single-sample failures from crashing pipelines
- Log warnings for skipped samples with counts (not every individual skip)
- Use `FileNotFoundError` with helpful messages when datasets/models are missing
- Always provide fallback paths (e.g., fall back to original dataset if augmented is missing)

## Performance
- Use `num_proc=1` for audio dataset `.map()` — librosa is not always fork-safe
- Use gradient checkpointing to maximise batch sizes
- Prefer `paged_adamw_8bit` optimizer for memory efficiency
- Set `dataloader_num_workers=8` on multi-core systems

## Testing Changes
- After modifying config, verify all downstream files (train, evaluate, demo, baseline) are consistent
- Model name changes in `config.py` must be reflected in `evaluate.py`, `run_baseline.py`, and `app.py`
- After augmentation changes, verify the augmented dataset loads correctly before training

## Security
- Never commit API keys, tokens, or `.env` files
- Use `report_to="none"` by default; only enable W&B explicitly
