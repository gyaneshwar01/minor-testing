"""
Transcript Cleaning Pipeline for Neplish ASR Dataset.

Parses the rich bracket annotation format used in the dataset and produces
clean target text suitable for Whisper fine-tuning.

Annotation Format Reference:
    [English/Nepali]   -> Code-switch marker, e.g. [digital/डिजिटल]
    [word/<unk>]        -> Uncertain transcription (transcriber's best guess)
    [<unk>/word]        -> Uncertain transcription (alternative form)
    [<unk>]             -> Completely unintelligible segment
    [<sil>]             -> Silence marker
    [<rep>/word]        -> Repetition / disfluency
    [<fin>]             -> End of utterance marker
    [<fil>/word]        -> Filler word
    [word1/word2]       -> Pronunciation variant (first = canonical)
"""

import re
import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core cleaning functions
# ---------------------------------------------------------------------------

def clean_transcript(text: str, keep_english: bool = True) -> str:
    """Clean a single transcript string by resolving all bracket annotations.

    Parameters
    ----------
    text : str
        Raw annotated transcript.
    keep_english : bool
        If True, code-switch markers ``[English/Nepali]`` resolve to the
        English word.  If False, they resolve to the Nepali transliteration.

    Returns
    -------
    str
        Cleaned transcript ready for ASR training.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Order matters: process more specific patterns first.

    # 1. [<fin>] -> remove (end-of-utterance metadata)
    text = re.sub(r"\[<fin>\]", "", text)

    # 2. [<sil>] -> remove (silence is not transcribed)
    text = re.sub(r"\[<sil>\]", "", text)

    # 3. [<unk>] standalone -> remove (completely unintelligible)
    text = re.sub(r"\[<unk>\]", "", text)

    # 4. [<rep>/word] -> keep the word (it was spoken, even if repeated)
    text = re.sub(r"\[<rep>/([^\]]+)\]", r"\1", text)

    # 5. [<fil>/word] -> keep the word (filler was spoken)
    text = re.sub(r"\[<fil>/([^\]]+)\]", r"\1", text)

    # 6. [<unk>/word] -> keep the word
    text = re.sub(r"\[<unk>/([^\]]+)\]", r"\1", text)

    # 7. [word/<unk>] -> keep the attempted word (trust transcriber's guess)
    text = re.sub(r"\[([^\]<]+)/<unk>\]", r"\1", text)

    # 8. [English/Nepali] code-switch markers
    #    Heuristic: if the first part is ASCII-only (Latin chars), treat it as
    #    English/Nepali code-switch.  Otherwise it's a pronunciation variant.
    def _resolve_code_switch(match: re.Match) -> str:
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        # Check if part1 is Latin-script (English)
        if re.fullmatch(r"[a-zA-Z0-9\s\-'\.]+", part1):
            return part1 if keep_english else part2
        # Otherwise treat as pronunciation variant -> keep first (canonical)
        return part1

    text = re.sub(r"\[([^\]]+)/([^\]]+)\]", _resolve_code_switch, text)

    # 9. Remove any remaining bare special tokens that might have leaked
    text = re.sub(r"<(?:unk|sil|rep|fin|fil)>", "", text)

    # 10. Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    keep_english: bool = True,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Apply transcript cleaning to an entire DataFrame.

    Adds a ``sentence`` column with the cleaned text and optionally removes
    rows that become empty after cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset DataFrame with a transcript column.
    text_col : str
        Name of the raw transcript column.
    keep_english : bool
        Passed through to :func:`clean_transcript`.
    drop_empty : bool
        If True, rows where the cleaned text is empty are dropped.

    Returns
    -------
    pd.DataFrame
        DataFrame with the new ``sentence`` column.
    """
    df = df.copy()
    df["sentence"] = df[text_col].apply(
        lambda t: clean_transcript(str(t), keep_english=keep_english)
    )

    if drop_empty:
        before = len(df)
        df = df[df["sentence"].str.len() > 0].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.info("Dropped %d rows with empty transcripts after cleaning.", dropped)

    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main(
    input_csv: Optional[str] = None,
    output_csv: Optional[str] = None,
    keep_english: bool = True,
) -> None:
    """Run the cleaning pipeline from the command line."""
    project_root = Path(__file__).resolve().parents[2]

    if input_csv is None:
        input_csv = str(project_root / "Dataset" / "combined_dataset.csv")
    if output_csv is None:
        output_csv = str(project_root / "Dataset" / "cleaned_dataset.csv")

    logger.info("Reading %s ...", input_csv)
    df = pd.read_csv(input_csv)
    logger.info("Loaded %d rows.", len(df))

    df = clean_dataframe(df, keep_english=keep_english)
    logger.info("After cleaning: %d rows.", len(df))

    df.to_csv(output_csv, index=False)
    logger.info("Saved cleaned dataset to %s", output_csv)

    # Quick summary
    has_english = df["sentence"].apply(lambda s: bool(re.search(r"[a-zA-Z]{2,}", s)))
    print(f"\n{'='*50}")
    print(f"Cleaning Summary")
    print(f"{'='*50}")
    print(f"  Total samples kept : {len(df)}")
    print(f"  Code-switched      : {has_english.sum()} ({has_english.mean():.1%})")
    print(f"  Pure Nepali         : {(~has_english).sum()} ({(~has_english).mean():.1%})")
    print(f"  Avg sentence length : {df['sentence'].str.len().mean():.1f} chars")
    print(f"  Output              : {output_csv}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Clean Neplish ASR transcripts.")
    parser.add_argument("--input", type=str, default=None, help="Input CSV path.")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path.")
    parser.add_argument(
        "--keep-nepali",
        action="store_true",
        help="Keep Nepali transliteration instead of English for code-switch markers.",
    )
    args = parser.parse_args()

    main(
        input_csv=args.input,
        output_csv=args.output,
        keep_english=not args.keep_nepali,
    )

