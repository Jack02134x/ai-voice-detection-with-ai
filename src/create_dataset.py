from pathlib import Path
import pandas as pd
import re

AUDIO_ROOT = Path("audio")
OUTPUT_FILE = Path("data/dataset.csv")

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
}


def youtube_group(filename):
    """
    Convert a snipped YouTube filename into the original-video group.

    Example:
        001_abc123_000.wav
        001_abc123_001.wav
        001_abc123_002.wav

    becomes:
        YouTube-AI::001_abc123
    """

    stem = Path(filename).stem

    # Remove the final clip number added by snip_4sec.py
    match = re.match(r"^(.+)_\d{3}$", stem)

    if match:
        original_name = match.group(1)
    else:
        original_name = stem

    return f"YouTube-AI::{original_name}"


def infer_metadata(path, label):
    """
    Infer dataset metadata from the file path/name.

    Returns:
        source, speaker, generator, language, group
    """

    parts = [part.strip().lower() for part in path.parts]
    filename = path.stem.lower()

    source = "unknown"
    speaker = ""
    generator = "unknown"
    language = "unknown"
    group = ""

    # ============================================================
    # HUMAN DATA
    # ============================================================

    if label == 0:

        if "garystafford" in parts:
            recording_id = filename.split("_part_", 1)[0]

            return (
                "GaryStafford",
                "",
                "",
                "en",
                f"GaryStafford::human::{recording_id}",
            )

        if "ljspeech" in parts:
            utterance_id = filename.split("_", 1)[0]

            return (
                "LJSpeech",
                "LJ",
                "",
                "en",
                f"LJSpeech::{utterance_id}",
            )

        if "crema-d" in parts or "cremad" in parts:
            speaker = filename.split("_", 1)[0]

            return (
                "CREMA-D",
                speaker,
                "",
                "en",
                f"CREMA-D::actor::{speaker}",
            )

        if "for-rerec" in parts or "for_rerec" in parts:
            recording_id = filename.split("_", 1)[0]

            return (
                "Fake-or-Real",
                "",
                "",
                "en",
                f"Fake-or-Real::human::{recording_id}",
            )

        return (
            "unknown",
            "",
            "",
            "unknown",
            f"unknown::human::{filename}",
        )

    # ============================================================
    # AI DATA
    # ============================================================

    if label == 1:

        # --------------------------------------------------------
        # GaryStafford synthetic voices
        # --------------------------------------------------------

        if "garystafford" in parts:

            generator_map = {
                "po": "Amazon Polly",
                "el": "ElevenLabs",
                "hg": "Kokoro",
                "hu": "Hume AI",
                "lv": "Luvvoice",
                "sp": "Speechify",
            }

            prefix = filename.split("_", 1)[0]
            generator = generator_map.get(prefix, "unknown")

            parts_after_prefix = filename.split("_", 1)

            if len(parts_after_prefix) > 1:
                recording_id = parts_after_prefix[1].split(
                    "_part_", 1
                )[0]
            else:
                recording_id = filename

            group = (
                f"GaryStafford::ai::{generator}::{recording_id}"
            )

            return (
                "GaryStafford",
                "",
                generator,
                "en",
                group,
            )

        # --------------------------------------------------------
        # YouTube AI dataset
        # --------------------------------------------------------

        # This catches folders such as:
        #
        # audio/ai/YouTube-AI/...
        #
        # and also:
        #
        # audio/ai/youtube/...
        #
        # The generator is intentionally unknown because we
        # don't want to pretend we know how each video was made.

        if "youtube-ai" in parts or "youtube_ai" in parts or "youtube" in parts:

            return (
                "YouTube-AI",
                "",
                "unknown",
                "unknown",
                youtube_group(path.name),
            )

        # --------------------------------------------------------
        # ASVspoof
        # --------------------------------------------------------

        if "asvspoof" in parts:

            return (
                "ASVspoof2021_DF",
                "",
                "unknown",
                "en",
                f"ASVspoof::{filename}",
            )

        # --------------------------------------------------------
        # WaveFake
        # --------------------------------------------------------

        if "wavefake" in parts:

            return (
                "WaveFake",
                "",
                "unknown",
                "unknown",
                f"WaveFake::{filename}",
            )

        # --------------------------------------------------------
        # Unknown AI dataset
        # --------------------------------------------------------

        return (
            "unknown",
            "",
            "unknown",
            "unknown",
            f"unknown::ai::{filename}",
        )

    # ============================================================
    # FALLBACK
    # ============================================================

    return (
        "unknown",
        "",
        "",
        "unknown",
        f"unknown::{filename}",
    )


def scan_directory(directory, label):
    rows = []

    if not directory.exists():
        print(f"WARNING: {directory} does not exist")
        return rows

    for path in directory.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        source, speaker, generator, language, group = infer_metadata(
            path,
            label,
        )

        rows.append(
            {
                "file": str(path),
                "label": label,
                "source": source,
                "speaker": speaker,
                "generator": generator,
                "language": language,
                "group": group,
            }
        )

    return rows


def main():

    human_dir = AUDIO_ROOT / "human"
    ai_dir = AUDIO_ROOT / "ai"

    print("Scanning audio files...")
    print()

    human_rows = scan_directory(
        human_dir,
        label=0,
    )

    ai_rows = scan_directory(
        ai_dir,
        label=1,
    )

    rows = human_rows + ai_rows

    df = pd.DataFrame(
        rows,
        columns=[
            "file",
            "label",
            "source",
            "speaker",
            "generator",
            "language",
            "group",
        ],
    )

    df = df.sort_values("file").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Human files: {len(human_rows)}")
    print(f"AI files:    {len(ai_rows)}")
    print(f"Total files: {len(df)}")
    print()

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print()

    print("Sources:")
    print(
        df["source"].value_counts(
            dropna=False
        )
    )

    print()

    print("Generators:")
    print(
        df["generator"]
        .replace("", "unknown")
        .value_counts()
    )

    print()

    print("Groups:")
    print(
        f"Unique groups: {df['group'].nunique()}"
    )

    print()

    print("File formats:")
    print(
        df["file"]
        .str.lower()
        .str.rsplit(".", n=1)
        .str[-1]
        .value_counts()
    )


if __name__ == "__main__":
    main()