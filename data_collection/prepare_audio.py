"""Cut recordings / videos into training clips under audio/.

Based on snip_vid_4sec.py. Changes: clips are 64,600 samples (~4.04 s, the
AASIST input size), they go into this project's audio/ folder, video files
work too, silent clips are skipped, files are decoded a minute at a time
so even multi-hour recaps barely use any RAM, and each file gives at most
--max-clips clips (default 1000), spread evenly over its whole length.

Usage:
  python data_collection/prepare_audio.py <file_or_directory> [...] --to ai/<name>
  python data_collection/prepare_audio.py <file_or_directory> [...] --to human/<name>
  python data_collection/prepare_audio.py long_recap.mp3 --to ai/<name> --max-clips 500
  python data_collection/prepare_audio.py stream.mp4 --to human/<name> --max-clips 0      # keep every clip

Examples:
  python data_collection/prepare_audio.py recap.mp3 --to ai/Manhwa_explained
      -> audio/ai/Manhwa_explained/recap/recap_000.wav ...
  python data_collection/prepare_audio.py ~/Downloads/manhwa_shit/ --to ai/Manhwa_explained
      -> audio/ai/Manhwa_explained/manhwa_shit/<sub folders>/<file>_000.wav ...

Directories are searched recursively. Exits with code 1 if any file failed.
Afterwards run: python build_local_csv.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchcodec.decoders import AudioDecoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # project root (for detector/)
from detector.config import LOCAL_AUDIO_ROOT  # noqa: E402


TARGET_SAMPLE_RATE = 16_000
CLIP_SAMPLES = 64_600                    # ~4.04 s, what AASIST takes as input
CLIP_DURATION = CLIP_SAMPLES / TARGET_SAMPLE_RATE
CHUNK_SECONDS = 60                       # decode this much at a time
SILENCE_DBFS = -50.0                     # clips quieter than this (RMS) are skipped

AUDIO_ROOT = LOCAL_AUDIO_ROOT / "audio"          # this project's audio/ folder

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
}


def get_audio_files(path):
    path = Path(path)

    if path.is_file():
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            return [path]

        print(f"Skipping unsupported file: {path}")
        return []

    if path.is_dir():
        files = sorted(
            file
            for file in path.rglob("*")
            if file.is_file()
            and file.suffix.lower() in AUDIO_EXTENSIONS
        )

        if not files:
            print(f"No supported audio files found in: {path}")

        return files

    print(f"Path does not exist: {path}")
    return []


def rms_dbfs(clip):
    return 20 * torch.log10(clip.pow(2).mean().sqrt() + 1e-12).item()


def snip_file(audio_path, output_dir, max_clips=0):
    audio_path = Path(audio_path)

    print()
    print("=" * 60)
    print(f"Input: {audio_path}")
    print("=" * 60)

    # Decoder resamples to 16 kHz and mixes down to mono while decoding.
    decoder = AudioDecoder(
        str(audio_path),
        sample_rate=TARGET_SAMPLE_RATE,
        num_channels=1,
    )

    duration = decoder.metadata.duration_seconds
    if duration:
        print(f"Duration: {duration:.2f} seconds ({duration / 60:.1f} min)")

    # With a limit, keep clip positions spread evenly over the whole file
    # (not just the first N), so one long video still shows all its variety.
    keep = None
    if max_clips and duration:
        total = int(duration // CLIP_DURATION)
        if total > max_clips:
            keep = set(np.linspace(0, total - 1, max_clips).round().astype(int).tolist())
            print(f"Limit: keeping {max_clips} of {total} clips, spread evenly over the file")
    elif max_clips:
        print(f"Limit: duration unknown, keeping the first {max_clips} clips")

    output_dir.mkdir(parents=True, exist_ok=True)

    buffer = torch.empty(1, 0)
    clip_index = 0
    clip_count = 0
    skipped = 0
    not_kept = 0
    start_seconds = 0.0

    while duration is None or start_seconds < duration:
        try:
            chunk = decoder.get_samples_played_in_range(
                start_seconds,
                start_seconds + CHUNK_SECONDS,
            ).data
        except (RuntimeError, ValueError, IndexError):
            break                                        # past the end

        if chunk.numel() == 0:
            break

        buffer = torch.cat([buffer, chunk], dim=1)
        start_seconds += CHUNK_SECONDS

        if max_clips and keep is None and clip_count >= max_clips:
            break                                        # unknown duration: first N only

        while buffer.shape[1] >= CLIP_SAMPLES:
            clip = buffer[:, :CLIP_SAMPLES]
            buffer = buffer[:, CLIP_SAMPLES:]

            if keep is not None and clip_index not in keep:
                not_kept += 1                            # over the limit, not printed
                clip_index += 1
                continue
            if max_clips and keep is None and clip_count >= max_clips:
                break

            start = clip_index * CLIP_DURATION
            end = start + CLIP_DURATION
            level = rms_dbfs(clip)

            if level < SILENCE_DBFS:
                print(
                    f"  [{clip_index:03d}] "
                    f"{start:.1f}s -> {end:.1f}s"
                    f"  skipped (silent, {level:.0f} dBFS)",
                    flush=True,
                )
                skipped += 1
                clip_index += 1
                continue

            output_name = f"{audio_path.stem}_{clip_index:03d}.wav"
            output_path = output_dir / output_name

            torchaudio.save(
                output_path,
                clip,
                TARGET_SAMPLE_RATE,
            )

            print(
                f"  [{clip_index:03d}] "
                f"{start:.1f}s -> {end:.1f}s"
                f"  {output_path.name}",
                flush=True,
            )

            clip_count += 1
            clip_index += 1

    print()
    print(f"Created: {clip_count} clips")

    if skipped:
        print(f"Skipped: {skipped} silent clips")

    if not_kept:
        print(f"Not kept: {not_kept} clips (over the --max-clips limit)")

    if clip_index == 0:
        print(f"Skipping: audio is shorter than {CLIP_DURATION:.2f} seconds.")

    if buffer.shape[1]:
        print(
            f"Ignored final {buffer.shape[1] / TARGET_SAMPLE_RATE:.2f}s "
            f"(shorter than {CLIP_DURATION:.2f} seconds)"
        )

    return clip_count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("inputs", nargs="+", help="files or directories")
    parser.add_argument(
        "--to",
        required=True,
        help="where under audio/ the clips go, e.g. ai/Manhwa_explained",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=1000,
        help="at most this many clips per file, spread over its length (0 = no limit)",
    )
    args = parser.parse_args()

    if args.to.split("/")[0] not in ("ai", "human"):
        parser.error("--to must start with ai/ or human/")

    output_root = AUDIO_ROOT / args.to
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_root}")
    print(f"Max clips per file: {args.max_clips or 'no limit'}")

    total_files = 0
    total_clips = 0
    errors = 0

    for input_path in args.inputs:
        input_path = Path(input_path).expanduser()

        print()
        print("#" * 60)
        print(f"Processing: {input_path}")
        print("#" * 60)

        files = get_audio_files(input_path)

        if not files:
            continue

        print(f"Found {len(files)} audio file(s).")

        for audio_file in files:
            try:
                # Each input gets its own directory.
                #
                # File:
                #   recap.mp3
                # -> <output_root>/recap/
                #
                # Directory:
                #   manhwa_shit/
                # -> <output_root>/manhwa_shit/
                #
                if input_path.is_file():
                    file_output_dir = output_root / input_path.stem
                else:
                    file_output_dir = (
                        output_root
                        / input_path.name
                        / audio_file.parent.relative_to(input_path)
                    )

                clips_created = snip_file(
                    audio_file,
                    file_output_dir,
                    args.max_clips,
                )

                total_files += 1
                total_clips += clips_created

            except KeyboardInterrupt:
                raise

            except Exception as e:
                errors += 1
                print()
                print(f"ERROR processing {audio_file}:")
                print(f"  {type(e).__name__}: {e}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Files processed: {total_files}")
    print(f"Clips created:   {total_clips}")
    print(f"Output:          {output_root}")
    if errors:
        print(f"Errors:          {errors} file(s) failed")
    print()
    print("Now run: python build_local_csv.py")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
