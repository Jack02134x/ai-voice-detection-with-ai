"""Stream training / validation / test audio straight from the Hugging Face Hub.

Nothing is written to disk: parquet shards are read with HTTP range requests
(one row group at a time, kept only in RAM) and audio bytes are decoded in
memory.
"""

import hashlib
import json
import math
import random
import time
from collections import defaultdict

from . import config  # noqa: F401  (sets HF_HOME before huggingface_hub is imported)

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfFileSystem
from torch.utils.data import IterableDataset, get_worker_info

from .audio_utils import decode_bytes, eval_clip, load_file, training_clip
from .augment import Channel
from .background import Backgrounds, noise_files
from .config import (BG_PROB, DATA_DIR, DEEPVOICE_HELDOUT, HUB_ORG, ITW_HELDOUT_BUCKETS, LOCAL_CSV, LOCAL_FOLDER_CAP,
                     LOCAL_WEIGHT, ONLINE_WEIGHT, SAMPLE_RATE, SHUFFLE_BUFFER, TRAIN_SOURCES)

MIN_SECONDS = 0.5
LOCAL = "Local"   # name of your own recordings as a data source
MIC = "Local+mic"  # the same held-out recordings through room + mic + codec (validation only)
SUBSET_CSV = DATA_DIR / "online_subset.csv"   # which online rows make up the fixed training pool


def decode_payload(payload):
    """Streamed rows carry audio bytes, local rows carry a file path."""
    return load_file(payload) if isinstance(payload, str) else decode_bytes(payload)


def local_rows(split):
    """[(path, label, folder)] from local_data.csv, or [] if it hasn't been built."""
    if LOCAL_WEIGHT <= 0 or not LOCAL_CSV.exists():
        return []
    import pandas as pd
    df = pd.read_csv(LOCAL_CSV, dtype={"file": str, "folder": str, "split": str})
    df = df[df["split"] == split]
    return list(zip(df["file"], df["label"].astype(int), df["folder"]))


def default_mix():
    """Share of training data per source: {name: weight}, incl. LOCAL."""
    total = sum(h + a for h, a, _ in TRAIN_SOURCES.values())
    mix = {name: ONLINE_WEIGHT * (h + a) / total for name, (h, a, _) in TRAIN_SOURCES.items()}
    mix[LOCAL] = LOCAL_WEIGHT
    return mix


def folder_weights(rows, folder_mult=None):
    """{label: {folder: sampling weight}}: sqrt(#files) capped at LOCAL_FOLDER_CAP,
    times an optional per-folder multiplier."""
    counts = defaultdict(lambda: defaultdict(int))
    for _, label, folder in rows:
        counts[label][folder] += 1
    return {label: {f: math.sqrt(min(n, LOCAL_FOLDER_CAP)) * (folder_mult or {}).get(f, 1.0)
                    for f, n in folders.items()}
            for label, folders in counts.items()}


def endless_local(rows, rng, folder_mult=None):
    """Your recordings: human/AI 50/50, then a folder weighted by sqrt(#files)
    (capped at LOCAL_FOLDER_CAP, times folder_mult) so small folders still show
    up without the big ones drowning them, then a random file."""
    weights_by_label = folder_weights(rows, folder_mult)
    classes = []
    for label in (0, 1):
        folders = defaultdict(list)
        for path, l, folder in rows:
            if l == label:
                folders[folder].append(path)
        names = [n for n in folders if weights_by_label[label][n] > 0]
        if names:
            classes.append((label, names, [weights_by_label[label][n] for n in names], folders))
    while True:
        label, names, weights, folders = rng.choice(classes)
        folder = rng.choices(names, weights=weights)[0]
        yield rng.choice(folders[folder]), label, folder


def retry(fn, what, tries=12, wait=10):
    """fn() again after short network outages (a phone hotspot's DNS drops out
    for a few seconds at a time); gives up after ~2 minutes."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == tries:
                raise
            print(f"[data] {what}: {type(e).__name__}: {e} -- retrying in {wait}s ({attempt}/{tries - 1})")
            time.sleep(wait)


def list_shards(fs, name):
    return retry(lambda: sorted(fs.glob(f"datasets/{HUB_ORG}/{name}/data/test-*.parquet")),
                 f"listing {name}")


def itw_speaker_heldout(notes: str) -> bool:
    """Speaker-disjoint split for In-the-Wild: same speaker -> same side."""
    speaker = json.loads(notes).get("speaker", "")
    return int(hashlib.md5(speaker.encode()).hexdigest(), 16) % ITW_HELDOUT_BUCKETS == 0


def deepvoice_side(notes: str):
    """'train', 'test' or None (dropped) for a DeepVoice row. Real speech goes by
    its speaker; an RVC conversion "a-to-b" only counts if a and b are on the same side."""
    stem = json.loads(notes).get("source_stem", "").lower()
    people = [stem.removesuffix("-original")] if stem.endswith("-original") else stem.split("-to-")
    held = {p in DEEPVOICE_HELDOUT for p in people}
    return None if len(held) > 1 else ("test" if held.pop() else "train")


# sources split by speaker inside their rows instead of by shard: notes -> 'train' / 'test' / None
ROW_SPLITS = {
    "InTheWild": lambda notes: "test" if itw_speaker_heldout(notes) else "train",
    "DeepVoice": deepvoice_side,
}


def iter_rows(fs, path, rng=None, keep=None):
    """Yield (audio_bytes, label, notes) from one parquet shard.

    With `rng`, row groups and rows inside each group are visited in random
    order. `keep(notes) -> bool` filters rows before their audio is fetched.
    """
    with fs.open(path, "rb", block_size=8 * 2**20) as f:
        pf = pq.ParquetFile(f)
        groups = list(range(pf.num_row_groups))
        if rng:
            rng.shuffle(groups)
        for g in groups:
            meta = pf.read_row_group(g, columns=["label", "notes"])
            labels = meta.column("label").to_pylist()
            notes = meta.column("notes").to_pylist()
            rows = [i for i in range(len(labels)) if keep is None or keep(notes[i])]
            if not rows:
                continue
            audio = pf.read_row_group(g, columns=["audio"]).column("audio").to_pylist()
            if rng:
                rng.shuffle(rows)
            for i in rows:
                yield audio[i]["bytes"], labels[i], notes[i]


def build_online_subset(seed=0):
    """Choose the fixed online training pool (TRAIN_SOURCES quotas) and write it to
    SUBSET_CSV: one line per (source, shard, row_group, row, label). Only the label /
    notes columns are read here, no audio. Row groups are drawn at random from ALL the
    training shards (many speakers / systems, not a few whole shards), taking rows
    until each class quota is met."""
    import pandas as pd
    fs, rng, out = HfFileSystem(), random.Random(seed), []
    for name, (want_h, want_a, n_val) in TRAIN_SOURCES.items():
        shards = list_shards(fs, name)
        shards = shards[:len(shards) - n_val]
        split = ROW_SPLITS.get(name)
        files = {path: retry(lambda: pq.ParquetFile(fs.open(path, "rb", block_size=2**20)), f"opening {path}")
                 for path in shards}
        pairs = [(path, g) for path, pf in files.items() for g in range(pf.num_row_groups)]
        rng.shuffle(pairs)
        got = {0: 0, 1: 0}
        want = {0: want_h, 1: want_a}
        for path, g in pairs:
            if got[0] >= want[0] and got[1] >= want[1]:
                break
            meta = retry(lambda: files[path].read_row_group(g, columns=["label", "notes"]), f"{name} labels")
            labels, notes = meta.column("label").to_pylist(), meta.column("notes").to_pylist()
            for i, (label, note) in enumerate(zip(labels, notes)):
                if got[label] < want[label] and (split is None or split(note) == "train"):
                    out.append((name, path, g, i, label))
                    got[label] += 1
        print(f"[subset] {name:12s} {got[0]:6d} human / {got[1]:6d} AI"
              + ("" if got == want else f"   (wanted {want[0]} / {want[1]}: not enough rows)"))
    df = pd.DataFrame(out, columns=["source", "shard", "row_group", "row", "label"])
    df.to_csv(SUBSET_CSV, index=False)
    print(f"[subset] {len(df)} online clips -> {SUBSET_CSV}")
    return df


def online_subset():
    """{source: [(shard, row_group, [rows])]} for the fixed online pool; (re)builds
    SUBSET_CSV when it is missing or the quotas in TRAIN_SOURCES changed."""
    import pandas as pd
    df = pd.read_csv(SUBSET_CSV) if SUBSET_CSV.exists() else None
    if df is not None:
        counts = df.groupby(["source", "label"]).size().to_dict()
        stale = set(df["source"]) != {n for n, (h, a, _) in TRAIN_SOURCES.items() if h + a} or any(
            counts.get((n, 0), 0) > h or counts.get((n, 1), 0) > a or
            (counts.get((n, 0), 0) < h and counts.get((n, 1), 0) < a)
            for n, (h, a, _) in TRAIN_SOURCES.items())
        if stale:
            print("[subset] TRAIN_SOURCES changed: rebuilding the online pool")
            df = None
    if df is None:
        df = build_online_subset()
    groups = defaultdict(list)
    for (name, shard, g), rows in df.groupby(["source", "shard", "row_group"])["row"]:
        groups[name].append((shard, int(g), sorted(rows.tolist())))
    return dict(groups)


def endless_subset(fs, groups, rng):
    """Cycle forever over the chosen rows of `groups` [(shard, row_group, rows)],
    one row group's audio in RAM at a time, in a fresh random order each pass."""
    while True:
        order = groups[:]
        rng.shuffle(order)
        for shard, g, rows in order:
            try:
                with fs.open(shard, "rb", block_size=8 * 2**20) as f:
                    t = pq.ParquetFile(f).read_row_group(g, columns=["audio", "label", "notes"])
                audio, labels, notes = (t.column(c).to_pylist() for c in ("audio", "label", "notes"))
            except Exception as e:  # network hiccup: move on
                print(f"[data] {shard.split('/')[-3]}: {type(e).__name__}: {e} -- skipping row group")
                time.sleep(2)
                continue
            rows = rows[:]
            rng.shuffle(rows)
            for i in rows:
                yield audio[i]["bytes"], labels[i], notes[i]


def endless_source(fs, shards, rng, keep=None):
    """Cycle through `shards` forever in a fresh random order each pass."""
    while True:
        order = shards[:]
        rng.shuffle(order)
        for path in order:
            try:
                yield from iter_rows(fs, path, rng, keep)
            except Exception as e:  # network hiccup: skip to the next shard
                print(f"[data] {path.split('/')[-3]}: {type(e).__name__}: {e} -- skipping shard")
                time.sleep(2)


class TrainStream(IterableDataset):
    """Infinite stream of (waveform[NB_SAMP], label) mixed from the fixed online
    pool (TRAIN_SOURCES, see online_subset) and your recordings. `mix` overrides source weights (see default_mix),
    `folder_mult` multiplies individual local folders' weights, a share
    `bg_prob` of clips (human and AI alike) gets a background sound mixed in,
    and `channel_scale` scales the room / device / codec augmentation (0 = off)."""

    def __init__(self, seed=0, mix=None, folder_mult=None, bg_prob=BG_PROB, channel_scale=1.0):
        self.seed = seed
        self.mix = mix or default_mix()
        self.subset = online_subset() if any(self.mix.get(n, 0) > 0 for n in TRAIN_SOURCES) else {}
        self.folder_mult = folder_mult or {}
        self.bg_prob = bg_prob
        self.channel_scale = channel_scale
        self.noise_paths = noise_files() if bg_prob > 0 else []

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(self.seed * 1009 + wid)
        np.random.seed((self.seed * 1009 + wid) % 2**32)
        fs = HfFileSystem()

        names, weights, sources = [], [], []
        for name in TRAIN_SOURCES:
            groups = self.subset.get(name, [])
            if self.mix.get(name, 0) <= 0 or not groups:
                continue
            mine = groups[wid::nw] or [groups[wid % len(groups)]]
            names.append(name)
            weights.append(self.mix[name])
            sources.append(endless_subset(fs, mine, rng))

        local = local_rows("train") if self.mix.get(LOCAL, 0) > 0 else []
        if local:
            names.append(LOCAL)
            weights.append(self.mix[LOCAL])
            sources.append(endless_local(local, rng, self.folder_mult))

        backgrounds = None
        if self.bg_prob > 0:
            babble = [path for path, label, _ in (local or local_rows("train")) if label == 0]
            backgrounds = Backgrounds(self.noise_paths, babble, seed=self.seed * 1009 + wid + 1)
        channel = Channel(seed=self.seed * 1009 + wid + 2, scale=self.channel_scale)

        buffer = []
        while True:
            src = rng.choices(range(len(sources)), weights=weights)[0]
            buffer.append(next(sources[src]))
            if len(buffer) < SHUFFLE_BUFFER:
                continue
            audio, label, _ = buffer.pop(rng.randrange(len(buffer)))
            try:
                wav = decode_payload(audio)
            except Exception:
                continue
            if len(wav) < MIN_SECONDS * SAMPLE_RATE:
                continue
            clip = channel.room(training_clip(wav))
            if backgrounds is not None and rng.random() < self.bg_prob:
                clip = backgrounds.mix(clip)
            yield torch.from_numpy(channel.transmit(clip)), label


def stream_eval_rows(name, limit=None, heldout_only=True):
    """Yield (audio_bytes, label, notes) deterministically from a source,
    restricted to its held-out part when it is also a training source.
    For LOCAL the audio is a file path and notes is the folder name."""
    if name == LOCAL:
        if limit and heldout_only:
            yield from balanced_local_val(limit)
        else:
            rows = local_rows("val" if heldout_only else "train")
            yield from rows[:limit] if limit else rows
        return
    fs = HfFileSystem()
    shards = list_shards(fs, name)
    keep = None
    if name in TRAIN_SOURCES and heldout_only:
        n_val = TRAIN_SOURCES[name][2]
        if n_val:
            shards = shards[-n_val:]
        elif name in ROW_SPLITS:
            keep = lambda n: ROW_SPLITS[name](n) == "test"  # noqa: E731
        else:
            raise ValueError(f"{name} is fully used for training; it has no held-out part")
    count = 0
    for path in shards:
        for row in iter_rows(fs, path, keep=keep):
            yield row
            count += 1
            if limit and count >= limit:
                return


def balanced_local_val(n):
    """Up to n held-out local clips: half human, half AI, folders taken round-robin."""
    rng = random.Random(0)
    picked = []
    for label in (0, 1):
        folders = defaultdict(list)
        for row in local_rows("val"):
            if row[1] == label:
                folders[row[2]].append(row)
        queues = [rng.sample(v, len(v)) for _, v in sorted(folders.items())]
        taken = []
        while queues and len(taken) < n // 2:
            for q in queues:
                if q and len(taken) < n // 2:
                    taken.append(q.pop())
            queues = [q for q in queues if q]
        picked += taken
    return picked


def validation_sources():
    return ["ASVspoof5", "InTheWild"] + ([LOCAL, MIC] if local_rows("val") else [])


def load_validation_set(per_source=600, sources=None):
    """Small fixed validation set held in RAM: {source: (waveforms, labels)}."""

    def collect(name):
        wavs, labels = [], []
        rows = balanced_local_val(per_source) if name in (LOCAL, MIC) else stream_eval_rows(name)
        effect = Channel(seed=0).full if name == MIC else None     # same "mic" every run
        for audio, label, _ in rows:
            try:
                wav = decode_payload(audio)
                if effect:
                    wav = effect(wav)
            except Exception:
                continue
            if len(wav) < MIN_SECONDS * SAMPLE_RATE:
                continue
            wavs.append(eval_clip(wav))
            labels.append(label)
            if len(wavs) >= per_source:
                break
        return wavs, labels

    out = {}
    for name in sources or validation_sources():
        # a network error part-way through starts this source over, so the set stays the same
        wavs, labels = retry(lambda: collect(name), f"validation set {name}")
        out[name] = (torch.from_numpy(np.stack(wavs)), torch.tensor(labels))
        print(f"[val] {name}: {len(labels)} clips, {sum(labels)} AI / {len(labels) - sum(labels)} human")
    return out
