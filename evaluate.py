"""Measure the detector on data it was never trained on.

    python evaluate.py --source SONAR                 # 8 modern TTS systems (streamed)
    python evaluate.py --source DeepVoice             # RVC voice conversion
    python evaluate.py --source InTheWild --limit 2000  # held-out speakers only
    python evaluate.py --source ASVspoof5 --limit 2000  # held-out shards only
    python evaluate.py --source Local --limit 600     # your held-out recordings, per folder
    python evaluate.py --folder benchmark             # your hard-case benchmark, per voice
    python evaluate.py --source Local --limit 600 --noise-snr 10   # same clips with background noise
    python evaluate.py --source Local --limit 600 --channel        # ... played into a mic, MP3/Opus'd
    python evaluate.py --folder benchmark --threshold 0.5          # a different decision threshold

A benchmark folder looks like  benchmark/ai/<voice>/...  and  benchmark/human/<voice>/...
(any audio/video files, any length, any depth), plus optionally
benchmark/noise/<kind>/... with silence / background noise that should come
out as "no speech". Keep it outside audio/ so it is never trained on.

Each file is scored exactly like detect.py does it: full length, sliding
4-second windows, averaged.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import detector.config  # noqa: F401  (sets HF_HOME first)
import numpy as np

from detector.audio_utils import load_file
from detector.config import LOCAL_CSV
from detector.data import LOCAL, decode_payload, stream_eval_rows
from detector.metrics import summarize
from detector.model import Detector

MEDIA_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".aac",
                    ".mp4", ".mkv", ".webm", ".mov"}


def benchmark_files(root):
    """[(path, label, voice)] from <root>/{ai,human,noise}/<voice>/... (noise: label None)"""
    rows = []
    for class_dir, label in (("human", 0), ("ai", 1), ("noise", None)):
        base = root / class_dir
        if not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if f.is_file() and f.suffix.lower() in MEDIA_EXTENSIONS:
                rel = f.relative_to(base).parts
                voice = rel[0] if len(rel) > 1 else "(loose files)"
                rows.append((f, label, f"{class_dir}/{voice}"))
    return rows


def warn_if_trained_on(rows):
    """Benchmark files that also appear (by file name) in the training index."""
    if not LOCAL_CSV.exists():
        return
    import pandas as pd
    df = pd.read_csv(LOCAL_CSV, usecols=["file", "split"])
    trained = {Path(f).name for f in df.loc[df["split"] == "train", "file"]}
    overlap = [r for r in rows if r[0].name in trained]
    if overlap:
        print(f"WARNING: {len(overlap)} benchmark file(s) share a name with training clips, "
              f"e.g. {overlap[0][0]} -- those scores are not a fair test.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="SONAR")
    p.add_argument("--folder", default=None, help="benchmark folder with ai/ and human/ inside")
    p.add_argument("--limit", type=int, default=None, help="max number of files")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--noise-snr", type=float, default=None,
                   help="mix a background under every file at this SNR in dB")
    p.add_argument("--noise-kind", default="synthetic", choices=["synthetic", "babble", "recorded"],
                   help="background for --noise-snr; babble/recorded come from the training pool")
    p.add_argument("--channel", action="store_true",
                   help="pass every file through room reverb + speaker/mic response + MP3/Opus first "
                        "(a 'played into a mic' test; works with --folder too)")
    p.add_argument("--threshold", type=float, default=None,
                   help="call it AI at P(AI) >= this (default: the one stored in the checkpoint)")
    args = p.parse_args()

    det = Detector(args.checkpoint, device=args.device, threshold=args.threshold)
    print(f"Model: {det.source}  |  threshold P(AI) >= {det.threshold:.3f}")

    if args.folder:
        root = Path(args.folder).expanduser()
        rows = benchmark_files(root)[:args.limit]
        if not rows:
            raise SystemExit(f"No files in {root}/ai, {root}/human or {root}/noise")
        warn_if_trained_on(rows)
        name = str(root)
        print(f"Benchmark {root}: {len(rows)} files")
        load = lambda path: load_file(str(path))  # noqa: E731
    else:
        rows = stream_eval_rows(args.source, limit=args.limit)
        name = args.source
        print(f"Streaming {args.source}...")
        load = decode_payload

    effects = []                       # applied to each waveform in this order
    if args.noise_snr is not None:
        from detector.background import Backgrounds, noise_files
        from detector.data import local_rows
        bg = Backgrounds(noise_files(), [r[0] for r in local_rows("train") if r[1] == 0],
                         seed=0, kinds={args.noise_kind: 1.0})
        name += f" + {args.noise_kind} background at {args.noise_snr:g} dB"
        effects.append(lambda wav: bg.mix(wav, args.noise_snr))
    if args.channel:
        from detector.augment import Channel
        name += " + room/mic/codec"
        effects.append(Channel(seed=0).full)

    if effects:
        def score(item):
            wav = load(item)
            for effect in effects:
                wav = effect(wav)
            return det.score_waveform(wav)
    elif args.folder:
        score = det.score_file             # reads long files in chunks
    else:
        score = lambda audio: det.score_waveform(decode_payload(audio))  # noqa: E731

    labels, scores, t0, done = [], [], time.time(), 0
    no_speech = {"human": 0, "ai": 0}
    per_group = defaultdict(list)    # group -> [(correct, P(AI) or nan, file)]
    for audio, label, notes in rows:
        try:
            r = score(audio)
        except Exception as e:
            print(f"  skip {audio if args.folder else ''}: {type(e).__name__}: {e}")
            continue
        done += 1
        s = r["ai_probability"]
        if label is None:                                 # benchmark/noise: should be "no speech"
            correct = r["no_speech"]
        elif r["no_speech"]:
            no_speech["ai" if label else "human"] += 1
            correct = False
        else:
            scores.append(s)
            labels.append(label)
            correct = (s >= det.threshold) == bool(label)
        if args.folder or args.source == LOCAL:
            per_group[notes].append((correct, s, audio))
        if done % 200 == 0:
            print(f"  {done} files  ({done / (time.time() - t0):.1f}/s)")

    if sum(no_speech.values()):
        print(f"\n  No speech found in {no_speech['human']} human and {no_speech['ai']} AI file(s) "
              f"-- left out of the numbers below.")
    if not labels:
        if per_group:
            for group, items in sorted(per_group.items()):
                print(f"    {group:28s} {np.mean([c for c, _, _ in items]):7.2%} detected as no speech"
                      f"   ({len(items)} files)")
            return
        raise SystemExit("Nothing could be scored.")
    r = summarize(np.array(labels), np.array(scores), det.threshold)
    print(f"\n{name}: {r['n']} files ({sum(labels)} AI / {len(labels) - sum(labels)} human)")
    if not np.isnan(r["eer"]):
        print(f"  EER                     {r['eer']:.2%}  (at P(AI) >= {r['eer_threshold']:.3f})")
    print(f"  Accuracy @ {det.threshold:.3f}        {r['accuracy']:.2%}")
    if not np.isnan(r["human_recall"]):
        print(f"  Human correctly human   {r['human_recall']:.2%}")
    if not np.isnan(r["ai_recall"]):
        print(f"  AI correctly AI         {r['ai_recall']:.2%}")

    if per_group:
        print("\n  Per folder:                    accuracy   mean P(AI)   (noise/: accuracy = 'no speech')")
        for group, items in sorted(per_group.items()):
            hits = [c for c, _, _ in items]
            ps = [s for _, s, _ in items if not np.isnan(s)]
            mean_p = f"{np.mean(ps):7.1%}" if ps else "    -  "
            print(f"    {group:28s} {np.mean(hits):7.2%}   {mean_p}   ({len(items)} files)")
        if args.folder:
            misses = [(g, s, f) for g, items in per_group.items() for c, s, f in items if not c]
            if misses:
                print("\n  Misclassified (worst first):")
                misses.sort(key=lambda m: -1 if np.isnan(m[1]) else abs(m[1] - det.threshold),
                            reverse=True)
                for g, s, f in misses[:15]:
                    what = "no speech" if np.isnan(s) else f"{s:6.1%} AI"
                    print(f"    {what:>9}  {g:24s} {f.name}")


if __name__ == "__main__":
    main()
