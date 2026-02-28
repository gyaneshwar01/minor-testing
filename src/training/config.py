"""
Training configuration for Neplish ASR Whisper fine-tuning.

All hyper-parameters and paths are centralised here so they can be
imported by both the standalone training script and the Colab notebook.

Uses QLoRA (4-bit NF4 quantization + LoRA) to fit whisper-large-v3 on
an RTX 4090 (24 GB VRAM) with optimised batch sizes.
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    """Whisper model settings."""

    model_name: str = "openai/whisper-large-v3"
    language: str = "ne"  # Nepali (ISO 639-1)
    task: str = "transcribe"


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit quantization settings for QLoRA."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"  # RTX 4090 (Ada Lovelace) supports bf16
    bnb_4bit_use_double_quant: bool = True      # nested quantization saves ~0.4 GB


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) settings for QLoRA/PEFT."""

    r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "SEQ_2_SEQ_LM"


@dataclass
class DataConfig:
    """Dataset paths and processing settings."""

    hf_dataset_dir: str = str(PROJECT_ROOT / "data" / "hf_dataset")
    hf_augmented_dataset_dir: str = str(PROJECT_ROOT / "data" / "hf_dataset_augmented")
    use_augmented: bool = True  # Set to True to train on augmented data
    max_audio_length_s: float = 30.0  # Whisper's max is 30s
    sampling_rate: int = 16_000


@dataclass
class TrainingConfig:
    """HuggingFace Trainer arguments.

    Batch sizes are tuned for RTX 4090 (24 GB VRAM) with whisper-large-v3 in 4-bit.
    Effective batch size = per_device * gradient_accumulation = 16 * 2 = 32.
    """

    output_dir: str = str(PROJECT_ROOT / "models" / "whisper-neplish")
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 16   # RTX 4090 24 GB can handle large batches
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2    # effective batch = 32
    learning_rate: float = 1e-4             # higher LR is typical for QLoRA
    warmup_steps: int = 300
    fp16: bool = False  # using bf16 instead (RTX 4090 native support)
    bf16: bool = True   # Ada Lovelace natively supports bfloat16
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    logging_steps: int = 50
    logging_first_step: bool = True
    remove_unused_columns: bool = False
    dataloader_num_workers: int = 8  # i9 14th gen + 64 GB RAM can feed GPU fast
    report_to: str = "none"  # Change to "wandb" if using W&B
    push_to_hub: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])

    # Gradient checkpointing saves memory at cost of ~20% slower training
    gradient_checkpointing: bool = True
    # Use 8-bit AdamW to further reduce optimizer memory footprint
    optim: str = "paged_adamw_8bit"


@dataclass
class NeplishASRConfig:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

