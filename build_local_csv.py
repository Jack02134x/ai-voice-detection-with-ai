"""Index your own recordings into data/local_data.csv.

    python build_local_csv.py

Scans <LOCAL_AUDIO_ROOT>/audio/{human,ai}/<folder>/ (same layout as the SIH
project) and reuses its data/sources.csv: folders with role "ignore" are
skipped and `group_regex` decides which clips came from the same recording.
Whole groups (videos / speakers) go to either train or val, never both, so
snippets of one video can't leak into validation.

Every clip is also checked with the speech detector (Silero VAD): clips with
less than MIN_SPEECH_SECONDS of speech (silence, background noise, music-only
intros) are left out, so the model is never taught "noise = human" or
"silence = AI". Results are cached in data/speech_cache.csv, so re-runs only
check new or changed files. --no-vad skips this check.

Re-run whenever you add or remove files.
"""

import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

from detector.config import (DATA_DIR, LOCAL_AUDIO_ROOT, LOCAL_CSV, LOCAL_EXCLUDE, LOCAL_SOURCES_CSV,
                    LOCAL_VAL_FRACTION, MIN_SPEECH_SECONDS, SAMPLE_RATE, VAD_PATH)

SPEECH_CACHE = DATA_DIR / "speech_cache.csv"

EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def folder_rules():
    if not LOCAL_SOURCES_CSV.exists():
        return {}
    df = pd.read_csv(LOCAL_SOURCES_CSV, dtype=str).fillna("")
    return {row["folder"]: row for row in df.to_dict("records")}


def group_of(folder, rel_path, regex):
    """Recording id a clip belongs to, e.g. the YouTube video it was cut from."""
    if regex:
        m = re.search(regex, rel_path)
        if m:
            return f"{folder}::{m.group(1) if m.groups() else m.group(0)}"
    stem = rel_path.rsplit(".", 1)[0]
    return f"{folder}::{re.sub(r'(_part)?_\d+$', '', stem)}"   # drop snippet numbering


def snippet_index(path):
    m = re.search(r"_(\d+)\.\w+$", path)
    return int(m.group(1)) if m else -1


def tail_split(items):
    """For folders that are basically one long recording: hold out the last
    LOCAL_VAL_FRACTION of each recording (highest snippet numbers), i.e. a later
    stretch of time rather than interleaved snippets."""
    by_group = defaultdict(list)
    for r in items:
        by_group[r["group"]].append(r)
    for group, rs in by_group.items():
        rs.sort(key=lambda r: (snippet_index(r["file"]), r["file"]))
        n_val = round(LOCAL_VAL_FRACTION * len(rs)) if len(rs) >= 10 else 0
        for i, r in enumerate(rs):
            r["split"] = "val" if i >= len(rs) - n_val else "train"
            if r["split"] == "val":
                r["group"] = f"{group}#tail"


def assign_splits(rows):
    """Per folder, move whole groups (videos / speakers) to val until
    ~LOCAL_VAL_FRACTION of its clips; folders with < 3 recordings use tail_split."""
    by_folder = defaultdict(list)
    for r in rows:
        by_folder[r["folder"]].append(r)
    for items in by_folder.values():
        sizes = Counter(r["group"] for r in items)
        if len(sizes) < 3:
            tail_split(items)
            continue
        groups = sorted(sizes, key=lambda g: hashlib.md5(g.encode()).hexdigest())
        target, n_val, val = LOCAL_VAL_FRACTION * len(items), 0, set()
        for g in groups[:-1]:                 # always keep at least one group for training
            if n_val >= target:
                break
            if n_val + sizes[g] > 1.5 * target:
                continue
            val.add(g)
            n_val += sizes[g]
        if n_val < 0.5 * target:      # a few huge recordings (e.g. hour-long recaps): split by time instead
            tail_split(items)
            continue
        for r in items:
            r["split"] = "val" if r["group"] in val else "train"


_vad = None


def _init_worker():
    global _vad
    import torch
    from detector.vad import SpeechDetector
    torch.set_num_threads(1)
    _vad = SpeechDetector()


def _speech_seconds(path):
    from detector.audio_utils import VAD_FRAME, load_file
    try:
        mask = _vad.speech_mask(load_file(path))
        return round(float(mask.sum() * VAD_FRAME / SAMPLE_RATE), 2)
    except Exception:
        return 0.0


def _save_cache(cache):
    pd.DataFrame([(f, m, sp) for (f, m), sp in cache.items() if os.path.exists(f)],
                 columns=["file", "mtime", "speech_s"]).to_csv(SPEECH_CACHE, index=False)


def add_speech_seconds(rows, workers):
    """rows[i]["speech_s"] = seconds of speech the VAD finds in that file.

    Runs in `workers` processes; results are cached by (file, modification time)
    and saved every few thousand clips, so an interrupted run loses little.
    """
    from multiprocessing import Pool

    cache = {}
    if SPEECH_CACHE.exists():
        c = pd.read_csv(SPEECH_CACHE)
        cache = dict(zip(zip(c["file"], c["mtime"]), c["speech_s"]))
    keys = {r["file"]: (r["file"], os.stat(r["file"]).st_mtime_ns) for r in rows}
    todo = [f for f, k in keys.items() if k not in cache]
    if todo:
        print(f"Checking {len(todo)} new clip(s) for speech with {workers} processes "
              f"(cached afterwards)...", flush=True)
        with Pool(workers, initializer=_init_worker) as pool:
            for i, (f, sec) in enumerate(zip(todo, pool.imap(_speech_seconds, todo, chunksize=32)), 1):
                cache[keys[f]] = sec
                if i % 2000 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
                if i % 5000 == 0:
                    _save_cache(cache)
        _save_cache(cache)
    for r in rows:
        r["speech_s"] = cache[keys[r["file"]]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-vad", action="store_true", help="don't drop clips without speech")
    p.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                   help="processes for the speech check")
    args = p.parse_args()

    audio_dir = LOCAL_AUDIO_ROOT / "audio"
    if not audio_dir.is_dir():
        sys.exit(f"Not found: {audio_dir}  (set LOCAL_AUDIO_ROOT)")
    rules = folder_rules()

    rows, skipped = [], []
    for class_dir, label in (("human", 0), ("ai", 1)):
        base = audio_dir / class_dir
        if not base.is_dir():
            continue
        for src in sorted(p for p in base.iterdir() if p.is_dir()):
            folder = f"{class_dir}/{src.name}"
            rule = rules.get(folder, {})
            if folder in LOCAL_EXCLUDE or rule.get("role") == "ignore":
                skipped.append(folder)
                continue
            for f in src.rglob("*"):
                if f.suffix.lower() in EXTENSIONS and f.is_file():
                    rel = f.relative_to(src).as_posix().replace("\\", "/")
                    rows.append({"file": str(f), "label": label, "folder": folder,
                                 "group": group_of(folder, rel, rule.get("group_regex", ""))})

    if not rows:
        sys.exit("No audio files found.")

    dropped = Counter()
    if not args.no_vad and VAD_PATH.exists():
        add_speech_seconds(rows, args.workers)
        dropped = Counter(r["folder"] for r in rows if r["speech_s"] < MIN_SPEECH_SECONDS)
        rows = [r for r in rows if r["speech_s"] >= MIN_SPEECH_SECONDS]

    assign_splits(rows)
    df = pd.DataFrame(rows)
    df.to_csv(LOCAL_CSV, index=False)

    summary = df.groupby(["label", "folder"]).agg(
        files=("file", "size"), recordings=("group", "nunique"),
        val_files=("split", lambda s: (s == "val").sum()))
    summary["no_speech_dropped"] = [dropped.get(f, 0) for _, f in summary.index]
    print(summary.to_string())
    print(f"\nHuman {int((df.label == 0).sum())} / AI {int((df.label == 1).sum())} files, "
          f"{int((df.split == 'val').sum())} held out for validation")
    if dropped:
        print(f"Left out {sum(dropped.values())} clips with no speech (silence / noise / music only)")
    if skipped:
        print(f"Skipped folders: {', '.join(skipped)}")
    print(f"Wrote {LOCAL_CSV}")


if __name__ == "__main__":
    main()
