"""
Fine-tune Whisper with QLoRA for Neplish (code-switched Nepali-English) ASR.

This script:
  1. Loads the pre-built HuggingFace dataset (train/val/test).
  2. Prepares Whisper feature-extractor and tokenizer.
  3. Loads whisper-large-v3 in 4-bit (NF4) via BitsAndBytes.
  4. Applies LoRA via PEFT on top of the quantized model (= QLoRA).
  5. Trains with the HuggingFace Seq2SeqTrainer.
  6. Evaluates on validation set using WER.

Usage (server):
    python -m src.training.train
"""

import logging
import os
from dataclasses import asdict
from functools import partial
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from src.training.config import NeplishASRConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------

class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for Whisper fine-tuning.

    Handles padding of both input features (mel spectrograms) and
    label token sequences.
    """

    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        # Separate input features and labels
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it is ignored by the loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the decoder_start_token if it was prepended
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Dataset preprocessing
# ---------------------------------------------------------------------------

def preprocess_function(
    examples: dict,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer,
    max_audio_length_s: float = 30.0,
) -> dict:
    """Extract mel features and tokenise the transcript."""
    audio = examples["audio"]

    # Extract mel spectrogram features
    input_features = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="np",
    ).input_features[0]

    # Tokenise the target text
    labels = tokenizer(examples["sentence"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred, tokenizer, metric_wer):
    """Compute WER on predictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric_wer.compute(predictions=pred_str, references=label_str)
    return {"wer": 100 * wer}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: NeplishASRConfig | None = None) -> None:
    """Run the full QLoRA training pipeline."""
    if config is None:
        config = NeplishASRConfig()

    # ------------------------------------------------------------------
    # 1. Load tokenizer, feature extractor, processor
    # ------------------------------------------------------------------
    logger.info("Loading model: %s", config.model.model_name)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config.model.model_name
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        config.model.model_name,
        language=config.model.language,
        task=config.model.task,
    )
    processor = WhisperProcessor.from_pretrained(
        config.model.model_name,
        language=config.model.language,
        task=config.model.task,
    )

    # ------------------------------------------------------------------
    # 2. Load model in 4-bit (QLoRA)
    # ------------------------------------------------------------------
    qcfg = config.quantization
    compute_dtype = torch.float16 if qcfg.bnb_4bit_compute_dtype == "float16" else torch.bfloat16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qcfg.load_in_4bit,
        bnb_4bit_quant_type=qcfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qcfg.bnb_4bit_use_double_quant,
    )

    logger.info(
        "Loading %s in 4-bit (%s, double_quant=%s) ...",
        config.model.model_name,
        qcfg.bnb_4bit_quant_type,
        qcfg.bnb_4bit_use_double_quant,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        config.model.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Whisper-specific: disable cache for gradient checkpointing compatibility
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Generation config
    model.generation_config.language = config.model.language
    model.generation_config.task = config.model.task
    model.generation_config.forced_decoder_ids = None

    # ------------------------------------------------------------------
    # 3. Prepare model for k-bit training and apply LoRA
    # ------------------------------------------------------------------
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config.training.gradient_checkpointing
    )

    logger.info("Applying LoRA (r=%d, alpha=%d) ...", config.lora.r, config.lora.lora_alpha)

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Load and preprocess dataset
    # ------------------------------------------------------------------
    # Use augmented dataset if available and configured
    dataset_dir = config.data.hf_dataset_dir
    if config.data.use_augmented:
        augmented_dir = config.data.hf_augmented_dataset_dir
        if Path(augmented_dir).exists():
            dataset_dir = augmented_dir
            logger.info("Using augmented dataset from %s", augmented_dir)
        else:
            logger.warning(
                "Augmented dataset not found at %s, falling back to %s",
                augmented_dir, dataset_dir,
            )

    logger.info("Loading dataset from %s", dataset_dir)
    dataset = load_from_disk(dataset_dir)

    logger.info("Preprocessing dataset (extracting features + tokenising) ...")
    prep_fn = partial(
        preprocess_function,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        max_audio_length_s=config.data.max_audio_length_s,
    )

    dataset = dataset.map(
        prep_fn,
        remove_columns=dataset["train"].column_names,
        num_proc=1,  # Audio decoding is not always safe with multiprocessing
    )

    # ------------------------------------------------------------------
    # 5. Set up trainer
    # ------------------------------------------------------------------
    metric_wer = evaluate.load("wer")

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.training.output_dir,
        num_train_epochs=config.training.num_train_epochs,
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        logging_steps=config.training.logging_steps,
        logging_first_step=config.training.logging_first_step,
        remove_unused_columns=config.training.remove_unused_columns,
        dataloader_num_workers=config.training.dataloader_num_workers,
        report_to=config.training.report_to,
        push_to_hub=config.training.push_to_hub,
        label_names=config.training.label_names,
        gradient_checkpointing=config.training.gradient_checkpointing,
        optim=config.training.optim,
        predict_with_generate=True,
        generation_max_length=225,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=partial(
            compute_metrics, tokenizer=tokenizer, metric_wer=metric_wer
        ),
        tokenizer=processor.feature_extractor,
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    logger.info("Starting training ...")
    trainer.train()

    # ------------------------------------------------------------------
    # 7. Save final model
    # ------------------------------------------------------------------
    final_dir = os.path.join(config.training.output_dir, "final")
    logger.info("Saving final model to %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    logger.info("Training complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    train()
