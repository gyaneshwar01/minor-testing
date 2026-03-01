"""
Gradio demo for Neplish ASR (Code-Switched Nepali-English Speech Recognition).

Features:
  - Microphone input for live transcription
  - Audio file upload
  - Side-by-side comparison: Vanilla Whisper vs Fine-tuned model
  - Code-switched word highlighting

Usage:
    python -m src.demo.app [--share]
"""

import argparse
import logging
import re
from pathlib import Path

import gradio as gr
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from peft import PeftModel

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Global model references (loaded once at startup)
_finetuned_model = None
_finetuned_processor = None
_vanilla_model = None
_vanilla_processor = None
_device = None


def get_device():
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def load_finetuned():
    """Load the fine-tuned QLoRA Whisper model (merged to fp16 for inference)."""
    global _finetuned_model, _finetuned_processor
    if _finetuned_model is not None:
        return _finetuned_model, _finetuned_processor

    model_dir = str(PROJECT_ROOT / "models" / "whisper-neplish" / "final")
    base_model_name = "openai/whisper-large-v3"
    device = get_device()

    logger.info("Loading fine-tuned model from %s ...", model_dir)

    try:
        _finetuned_processor = WhisperProcessor.from_pretrained(model_dir)
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch.float16,
        )
        _finetuned_model = PeftModel.from_pretrained(base_model, model_dir)
        _finetuned_model = _finetuned_model.merge_and_unload()
        _finetuned_model.to(device).eval()
        logger.info("Fine-tuned model loaded successfully.")
    except Exception as e:
        logger.warning("Could not load fine-tuned model: %s. Using vanilla as fallback.", e)
        _finetuned_model, _finetuned_processor = load_vanilla()

    return _finetuned_model, _finetuned_processor


def load_vanilla():
    """Load vanilla Whisper model."""
    global _vanilla_model, _vanilla_processor
    if _vanilla_model is not None:
        return _vanilla_model, _vanilla_processor

    model_name = "openai/whisper-large-v3"
    device = get_device()

    logger.info("Loading vanilla Whisper (%s) ...", model_name)
    _vanilla_processor = WhisperProcessor.from_pretrained(
        model_name, language="ne", task="transcribe"
    )
    _vanilla_model = WhisperForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16,
    )
    _vanilla_model.to(device).eval()
    logger.info("Vanilla Whisper loaded.")

    return _vanilla_model, _vanilla_processor


def transcribe(audio_input, model, processor) -> str:
    """Transcribe audio using a Whisper model."""
    if audio_input is None:
        return "No audio provided."

    device = get_device()

    # Handle both (sample_rate, array) tuples and file paths
    if isinstance(audio_input, tuple):
        sr, audio_array = audio_input
        # Convert to float32 and normalise
        audio_array = audio_array.astype(np.float32)
        if audio_array.max() > 1.0:
            audio_array = audio_array / np.iinfo(np.int16).max
        # Convert stereo to mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
    elif isinstance(audio_input, str):
        import librosa
        audio_array, sr = librosa.load(audio_input, sr=16000)
    else:
        return "Unsupported audio format."

    # Resample to 16kHz if needed
    if sr != 16000:
        import librosa
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    input_features = processor.feature_extractor(
        audio_array, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device=device, dtype=model.dtype)

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ne", task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            forced_decoder_ids=forced_decoder_ids,
            max_new_tokens=225,
        )

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return text.strip()


def highlight_english(text: str) -> str:
    """Wrap English words in bold markers for display."""
    def _bold_match(m):
        return f"**{m.group(0)}**"
    return re.sub(r"[a-zA-Z]{2,}", _bold_match, text)


# ---------------------------------------------------------------------------
# Gradio interface functions
# ---------------------------------------------------------------------------

def transcribe_finetuned(audio):
    """Transcribe using the fine-tuned model (generator for live progress)."""
    if audio is None:
        yield "No audio provided."
        return

    if _finetuned_model is None:
        yield "⏳ Loading fine-tuned model (first run only, ~30s)..."
    else:
        yield "🎙️ Transcribing..."
    model, processor = load_finetuned()

    yield "🎙️ Transcribing audio..."
    text = transcribe(audio, model, processor)
    yield highlight_english(text)


def transcribe_vanilla(audio):
    """Transcribe using vanilla Whisper (generator for live progress)."""
    if audio is None:
        yield "No audio provided."
        return

    if _vanilla_model is None:
        yield "⏳ Loading vanilla Whisper (first run only, ~30s)..."
    else:
        yield "🎙️ Transcribing..."
    model, processor = load_vanilla()

    yield "🎙️ Transcribing audio..."
    text = transcribe(audio, model, processor)
    yield highlight_english(text)


def transcribe_both(audio):
    """Transcribe with both models and return side-by-side results (generator for live progress)."""
    if audio is None:
        yield "No audio provided.", "No audio provided."
        return

    # Step 1: Load fine-tuned model
    if _finetuned_model is None:
        yield "⏳ Loading fine-tuned model...", "⏳ Waiting..."
    else:
        yield "🎙️ Preparing...", "⏳ Waiting..."
    ft_model, ft_proc = load_finetuned()

    # Step 2: Load vanilla model
    if _vanilla_model is None:
        yield "⏳ Loading vanilla Whisper...", "✅ Fine-tuned model ready"
    else:
        yield "🎙️ Preparing...", "✅ Fine-tuned model ready"
    van_model, van_proc = load_vanilla()

    # Step 3: Transcribe with fine-tuned
    yield "⏳ Waiting...", "🎙️ Transcribing (fine-tuned)..."
    ft_text = transcribe(audio, ft_model, ft_proc)
    ft_highlighted = highlight_english(ft_text)

    # Step 4: Transcribe with vanilla
    yield "🎙️ Transcribing (baseline)...", ft_highlighted
    van_text = transcribe(audio, van_model, van_proc)
    van_highlighted = highlight_english(van_text)

    yield van_highlighted, ft_highlighted


def build_demo() -> gr.Blocks:
    """Build the Gradio Blocks UI."""
    with gr.Blocks(
        title="Neplish ASR",
    ) as demo:
        gr.Markdown(
            """
            # Neplish ASR - Code-Switched Nepali-English Speech Recognition

            Transcribe **Neplish** (code-switched Nepali-English) speech using a
            fine-tuned Whisper model. Compare results against the vanilla Whisper baseline.

            English words in the output are highlighted in **bold**.
            """
        )

        with gr.Tabs():
            # --- Tab 1: Single model transcription ---
            with gr.TabItem("Transcribe"):
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="Input Audio",
                            type="numpy",
                            sources=["microphone", "upload"],
                        )
                        transcribe_btn = gr.Button("Transcribe", variant="primary")
                    with gr.Column():
                        output_text = gr.Markdown(label="Transcription")

                transcribe_btn.click(
                    fn=transcribe_finetuned,
                    inputs=audio_input,
                    outputs=output_text,
                )

            # --- Tab 2: Side-by-side comparison ---
            with gr.TabItem("Compare Models"):
                with gr.Row():
                    with gr.Column():
                        audio_compare = gr.Audio(
                            label="Input Audio",
                            type="numpy",
                            sources=["microphone", "upload"],
                        )
                        compare_btn = gr.Button("Compare", variant="primary")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Vanilla Whisper (Baseline)")
                        vanilla_output = gr.Markdown()
                    with gr.Column():
                        gr.Markdown("### Fine-tuned Whisper (Neplish)")
                        finetuned_output = gr.Markdown()

                compare_btn.click(
                    fn=transcribe_both,
                    inputs=audio_compare,
                    outputs=[vanilla_output, finetuned_output],
                )

            # --- Tab 3: About ---
            with gr.TabItem("About"):
                gr.Markdown(
                    """
                    ## About This Project

                    **Neplish ASR** is a speech recognition system built to handle
                    code-switched Nepali-English speech - a common phenomenon in
                    everyday Nepali conversations where English words and phrases
                    are naturally mixed in.

                    ### Technical Details

                    - **Base Model**: OpenAI Whisper (Large-v3) - 1.5B parameters
                    - **Fine-tuning**: QLoRA (4-bit NF4 quantization + LoRA) via PEFT
                    - **GPU**: NVIDIA RTX 4090 24 GB
                    - **Dataset**: ~6,800 annotated utterances (~20 hours) + augmented
                    - **Languages**: Nepali + English (code-switched)

                    ### Metrics

                    The model is evaluated on:
                    - **WER** (Word Error Rate) - overall transcription accuracy
                    - **CER** (Character Error Rate) - character-level accuracy
                    - **Code-Switch WER** - accuracy on code-switched utterances
                    - **English Token F1** - precision/recall for English words
                    """
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Neplish ASR Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on.")
    args = parser.parse_args()

    demo = build_demo()
    demo.launch(share=args.share, server_port=args.port, theme=gr.themes.Soft())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

