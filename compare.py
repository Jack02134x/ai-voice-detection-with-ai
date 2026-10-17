"""Compare checkpoints on every test at once and suggest a decision threshold for each.

    python compare.py models/checkpoints/v3/best.pt models/checkpoints/v4/best.pt
    python compare.py v4/best.pt v5/best.pt --tests local local-mic local-replay
    python compare.py ... --reuse      # re-print from saved scores instead of re-scoring

Every dataset is read once and scored by all checkpoints, so online sets are streamed
only once. Scores go to models/checkpoints/compare/<test>.npz.

Tests: local (your held-out clips), local-noise (+ 10 dB synthetic noise), local-mic
(room + mic + MP3/Opus, a different random room than validation uses), local-replay
(phone speaker -> room -> laptop mic), bench / bench-replay (benchmark/), sonar,
deepvoice (held-out people only).

The suggested threshold maximises the mean balanced accuracy over the tests that have
both classes (local, local-noise, local-mic, local-replay, sonar, deepvoice).
"""

import argparse
import time
from pathlib import Path

import detector.config  # noqa: F401  (sets HF_HOME first)
import numpy as np

from detector.audio_utils import load_file
from detector.config import CHECKPOINT_DIR, ROOT
from detector.metrics import compute_eer

OUT = CHECKPOINT_DIR / "compare"
TESTS = ["local", "local-noise", "local-mic", "local-replay", "bench", "bench-replay", "sonar", "deepvoice"]
THRESHOLDS = np.round(np.arange(0.15, 0.71, 0.05), 2)


def resolve(ckpt):
    p = Path(ckpt)
    return p if p.exists() else CHECKPOINT_DIR / ckpt


def rows_for(test):
    from evaluate import benchmark_files
    from detector.data import balanced_local_val, decode_payload, stream_eval_rows
    if test.startswith("local"):
        return balanced_local_val(600), decode_payload
    if test.startswith("bench"):
        return benchmark_files(ROOT / "benchmark"), lambda p: load_file(str(p))
    return stream_eval_rows({"sonar": "SONAR", "deepvoice": "DeepVoice"}[test]), decode_payload


def effect_for(test):
    from detector.augment import Channel
    if test.endswith("-mic"):
        return Channel(seed=1).full
    if test.endswith("-replay"):
        return Channel(seed=1).replay
    if test.endswith("-noise"):
        from detector.background import Backgrounds, noise_files
        from detector.data import local_rows
        bg = Backgrounds(noise_files(), [r[0] for r in local_rows("train") if r[1] == 0],
                         seed=0, kinds={"synthetic": 1.0})
        return lambda w: bg.mix(w, 10)
    return lambda w: w


def score_test(test, dets):
    rows, load = rows_for(test)
    fx = effect_for(test)
    labels, notes, scores, t0 = [], [], [[] for _ in dets], time.time()
    for i, (audio, label, note) in enumerate(rows):
        try:
            wav = fx(load(audio))
            res = [d.score_waveform(wav)["ai_probability"] for d in dets]
        except Exception as e:
            print(f"  skip: {type(e).__name__}: {str(e)[:80]}")
            continue
        labels.append(-1 if label is None else label)
        notes.append(str(note))
        for s, r in zip(scores, res):
            s.append(r)
        if (i + 1) % 250 == 0:
            print(f"  {test}: {i + 1} files ({(i + 1) / (time.time() - t0):.1f}/s)", flush=True)
    return np.array(labels), np.array(notes), np.array(scores)


def recalls(y, s, t):
    h = (s[y == 0] < t).mean() if (y == 0).any() else np.nan
    a = (s[y == 1] >= t).mean() if (y == 1).any() else np.nan
    return h, a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--tests", nargs="+", default=TESTS, choices=TESTS)
    p.add_argument("--reuse", action="store_true", help="use saved scores when they cover these checkpoints")
    args = p.parse_args()
    ckpts = [resolve(c) for c in args.checkpoints]
    names = [f"{c.parent.name}/{c.stem}" for c in ckpts]
    OUT.mkdir(parents=True, exist_ok=True)

    dets = None
    results = {}
    for test in args.tests:
        path = OUT / f"{test}.npz"
        if args.reuse and path.exists():
            d = np.load(path)
            if all(n in list(d["names"]) for n in names):
                idx = [list(d["names"]).index(n) for n in names]
                results[test] = (d["labels"], d["scores"][idx])
                continue
        if dets is None:
            from detector.model import Detector
            dets = [Detector(str(c)) for c in ckpts]
            for n, d in zip(names, dets):
                print(f"{n}: step {d.source}, stored threshold {d.threshold:.3f}")
        print(f"\n== scoring {test} ...", flush=True)
        y, notes, s = score_test(test, dets)
        np.savez(path, labels=y, notes=notes, scores=s, names=np.array(names))
        results[test] = (y, s)

    print("\n" + "=" * 100)
    for test, (y, S) in results.items():
        print(f"\n{test}: {(y == 1).sum()} AI / {(y == 0).sum()} human")
        for n, s in zip(names, S):
            ok = ~np.isnan(s) & (y >= 0)
            yy, ss = y[ok], s[ok]
            eer = compute_eer(yy, ss) if len(set(yy)) == 2 else (np.nan, np.nan)
            cells = []
            for t in (0.3, 0.4, 0.5):
                h, a = recalls(yy, ss, t)
                cells.append(f"@{t}: {h:4.0%}/{a:4.0%}".replace("nan%", "  - "))
            print(f"  {n:16s} EER {eer[0]:6.1%} (at {eer[1]:.2f})  no-speech {(~ok).sum():3d}   "
                  + "   ".join(cells) + "   [human/AI right]")

    print("\nSuggested threshold (max mean balanced accuracy over tests with both classes):")
    both = [t for t, (y, _) in results.items() if (y == 0).any() and (y == 1).any()]
    for i, n in enumerate(names):
        best = None
        for t in THRESHOLDS:
            bal = []
            for test in both:
                y, S = results[test]
                ok = ~np.isnan(S[i]) & (y >= 0)
                h, a = recalls(y[ok], S[i][ok], t)
                bal.append((h + a) / 2)
            if best is None or np.mean(bal) > best[1]:
                best = (t, np.mean(bal), dict(zip(both, bal)))
        t, m, per = best
        print(f"  {n:16s} {t:.2f}  -> mean balanced accuracy {m:.1%}  ("
              + ", ".join(f"{k} {v:.0%}" for k, v in per.items()) + ")")


if __name__ == "__main__":
    main()
