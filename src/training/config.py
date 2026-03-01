"""
Training configuration for Neplish ASR Whisper fine-tuning.

All hyper-parameters and paths are centralised here so they can be
imported by both the standalone training script and the Colab notebook.

Uses QLoRA (4-bit NF4 quantization + LoRA) to fit whisper-large-v3 on
an RTX 4090 24 GB GPU.
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
    bnb_4bit_compute_dtype: str = "bfloat16"  # native on Ada Lovelace (4090)
    bnb_4bit_use_double_quant: bool = True   # nested quantization saves ~0.4 GB


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
    external_data_dir: str = str(PROJECT_ROOT / "data" / "external")
    max_audio_length_s: float = 30.0  # Whisper's max is 30s
    sampling_rate: int = 16_000
    use_augmented: bool = True  # Whether to use augmented dataset for training


@dataclass
class TrainingConfig:
    """HuggingFace Trainer arguments.

    Batch sizes are tuned for an RTX 4090 24 GB GPU with whisper-large-v3 in 4-bit.
    Effective batch size = per_device * gradient_accumulation = 16 * 2 = 32.
    """

    output_dir: str = str(PROJECT_ROOT / "models" / "whisper-neplish")
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 16   # RTX 4090 24 GB has plenty of room
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2    # effective batch = 32
    learning_rate: float = 5e-5             # slightly lower LR for larger model
    warmup_steps: int = 500
    fp16: bool = False  # use bf16 instead on Ada Lovelace
    bf16: bool = True   # native bf16 on RTX 4090
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
    dataloader_num_workers: int = 4
    report_to: str = "none"  # Change to "wandb" if using W&B
    push_to_hub: bool = False
    label_names: list[str] = field(default_factory=lambda: ["labels"])

    # Gradient checkpointing disabled -- 24 GB is enough for large-v3 in 4-bit
    # Re-enable if OOM occurs (costs ~20% speed)
    gradient_checkpointing: bool = False
    # 8-bit paged AdamW keeps optimizer states compact
    optim: str = "paged_adamw_8bit"


@dataclass
class NeplishASRConfig:
    """Top-level configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

