"""
Training configuration for Neplish ASR Whisper fine-tuning.

All hyper-parameters and paths are centralised here so they can be
imported by both the standalone training script and the Colab notebook.

Uses QLoRA (4-bit NF4 quantization + LoRA) to fit whisper-medium on
a 6 GB GPU (e.g. RTX 4050 Laptop).
"""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    """Whisper model settings."""

    model_name: str = "openai/whisper-medium"
    language: str = "ne"  # Nepali (ISO 639-1)
    task: str = "transcribe"


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit quantization settings for QLoRA."""

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"  # "bfloat16" on Ampere+
    bnb_4bit_use_double_quant: bool = True   # nested quantization saves ~0.4 GB


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) settings for QLoRA/PEFT."""

    r: int = 16
    lora_alpha: int = 32
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
    max_audio_length_s: float = 30.0  # Whisper's max is 30s
    sampling_rate: int = 16_000


@dataclass
class TrainingConfig:
    """HuggingFace Trainer arguments.

    Batch sizes are tuned for a 6 GB GPU with whisper-medium in 4-bit.
    Effective batch size = per_device * gradient_accumulation = 4 * 4 = 16.
    """

    output_dir: str = str(PROJECT_ROOT / "models" / "whisper-neplish")
    num_train_epochs: int = 8
    per_device_train_batch_size: int = 4   # lowered for 6 GB VRAM
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4   # keeps effective batch = 16
    learning_rate: float = 1e-4            # higher LR is typical for QLoRA
    warmup_steps: int = 200
    fp16: bool = True  # Set to False if no GPU or using bf16
    bf16: bool = False
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

