"""Check whether recordings are a human voice or AI-generated.

    python detect.py some_audio.wav
    python detect.py podcast.mp3 --timeline         # also show every 4-second window
    python detect.py ~/Music/clips/ other.mp4       # many files / folders -> one line each
    python detect.py file.wav --device cpu          # leave the GPU to a running training

Any audio or video format works, and files of any length (decoded 10 minutes
at a time).
"""

import argparse
import sys
import time
from pathlib import Path

import detector.config  # noqa: F401  (sets HF_HOME first)

from detector.config import NB_SAMP, SAMPLE_RATE
from detector.model import Detector

RESET, BOLD, RED, GREEN, GRAY, YELLOW = ("\033[0m", "\033[1m", "\033[91m", "\033[92m",
                                       "\033[90m", "\033[93m")
MEDIA_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac",
                    ".mp4", ".mkv", ".webm", ".mov"}


def bar(p, width=30):
    filled = round(p * width)
    return "█" * filled + GRAY + "░" * (width - filled) + RESET


def hms(t):
    h, rem = divmod(int(t), 3600)
    return f"{h}:{rem // 60:02d}:{rem % 60:02d}" if h else f"{rem // 60:02d}:{t % 60:04.1f}"


def media_files(inputs):
    for item in inputs:
        path = Path(item).expanduser()
        if path.is_dir():
            yield from sorted(f for f in path.rglob("*")
                              if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS)
        elif path.is_file():
            yield path
        else:
            print(f"skip {path}: not found")


def print_detail(det, path, r, elapsed, timeline):
    probs, starts = r["window_probs"], r["window_starts"]
    win = NB_SAMP / SAMPLE_RATE
    print(f"\n{BOLD}File{RESET}      {path}")
    print(f"{BOLD}Duration{RESET}  {hms(r['duration'])}  ->  {len(probs)} speech windows "
          f"of {win:.2f} s  ({elapsed:.1f} s)")
    if r["no_speech"]:
        print(f"\n  {BOLD}Verdict: {YELLOW}NO SPEECH{RESET}  (only silence / background noise)\n")
        return
    if timeline:
        print()
        for s, p in zip(starts, probs):
            color = RED if p >= det.threshold else GREEN
            print(f"  {hms(s)}-{hms(s + win)}  {bar(p)}  {color}{p:6.1%} AI{RESET}")
    ai_windows = int((probs >= det.threshold).sum())
    print(f"\n  AI windows      {ai_windows}/{len(probs)}")
    print(f"  P(AI) overall   {bar(r['ai_probability'])} {r['ai_probability']:.1%}"
          f"   (threshold {det.threshold:.1%})")
    verdict = f"{RED}AI-GENERATED" if r["is_ai"] else f"{GREEN}HUMAN"
    print(f"\n  {BOLD}Verdict: {verdict}{RESET}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="audio/video files or folders")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--timeline", action="store_true", help="print every window's score")
    p.add_argument("--threshold", type=float, default=None,
                   help="call it AI at P(AI) >= this (default: the one stored in the checkpoint)")
    args = p.parse_args()

    files = list(media_files(args.inputs))
    if not files:
        sys.exit("No audio/video files found.")

    det = Detector(args.checkpoint, device=args.device, threshold=args.threshold)
    print(f"{BOLD}Model{RESET}     {det.source} on {det.device}"
          f"{'  + speech detector' if det.vad else ''}")

    detailed = len(files) == 1 or args.timeline
    results = []
    for path in files:
        t0 = time.time()

        def progress(done):
            if done >= 600:   # only worth showing for long files
                print(f"\r  {path.name[:50]}: {hms(done)} analysed...", end="", flush=True)

        try:
            r = det.score_file(path, progress=progress)
        except Exception as e:
            print(f"\r  {path.name}: ERROR {type(e).__name__}: {e}")
            continue
        print("\r" + " " * 80 + "\r", end="")
        if detailed:
            print_detail(det, path, r, time.time() - t0, args.timeline)
        elif r["no_speech"]:
            print(f"  {YELLOW}NO SPEECH{RESET}            {hms(r['duration']):>9}  {path.name}")
        else:
            color, label = (RED, "AI   ") if r["is_ai"] else (GREEN, "HUMAN")
            print(f"  {color}{label}{RESET} {r['ai_probability']:6.1%} AI  "
                  f"{hms(r['duration']):>9}  {path.name}")
        results.append("none" if r["no_speech"] else "ai" if r["is_ai"] else "human")

    if len(results) > 1:
        print(f"\n{len(results)} files: {results.count('ai')} AI-generated, "
              f"{results.count('human')} human, {results.count('none')} no speech  "
              f"(threshold {det.threshold:.1%})")


if __name__ == "__main__":
    main()
