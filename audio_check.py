"""Prove the Android app's Java preprocessing matches detector/audio_utils.py.

    python audio_check.py                       # a few local clips + benchmark files

For each 16 kHz test file it runs the Java Preprocess (desktop JVM) and the Python
sliding_windows (energy path) and checks the windows are identical, then runs both sets
of windows through aasist.onnx and checks the per-window P(AI) agree. Resampling is not
tested here (Java uses the phone decoder); all test files are already 16 kHz.
"""

import struct
import subprocess
import tempfile
from pathlib import Path

import detector.config  # noqa: F401
import numpy as np

from detector.audio_utils import load_file, sliding_windows
from detector.config import NB_SAMP, ROOT

ANDROID = ROOT / "android"
ASSETS = ANDROID / "assets"


def compile_java():
    out = Path(tempfile.mkdtemp(prefix="voicecheck_"))
    srcs = [str(ANDROID / "src/com/voicecheck/Preprocess.java"), str(ANDROID / "test/CheckMain.java")]
    subprocess.run(["javac", "-d", str(out), *srcs], check=True)
    return out


def java_windows(classes, wav):
    with tempfile.NamedTemporaryFile(suffix=".f32") as fin, tempfile.NamedTemporaryFile(suffix=".f32") as fout:
        fin.write(wav.astype("<f4").tobytes())
        fin.flush()
        n = int(subprocess.run(["java", "-cp", str(classes), "com.voicecheck.CheckMain", fin.name, fout.name],
                               capture_output=True, text=True, check=True).stdout.strip())
        data = Path(fout.name).read_bytes()
    return np.frombuffer(data, "<f4").reshape(n, NB_SAMP) if n else np.empty((0, NB_SAMP), np.float32)


def main():
    import onnxruntime as ort
    import pandas as pd
    classes = compile_java()
    sess = ort.InferenceSession(str(ASSETS / "aasist.onnx"), providers=["CPUExecutionProvider"])

    df = pd.read_csv(detector.config.LOCAL_CSV)
    files = df[df.folder == "human/vishesh"].file.tolist()[:3]
    files += df[df.folder.str.startswith("ai/")].file.tolist()[:3]
    files += [str(f) for f in sorted((ROOT / "benchmark").rglob("*.wav"))[100:103]]

    worst_win, worst_score, worst_n = 0.0, 0.0, 0
    for f in files:
        wav = load_file(f)
        jw = java_windows(classes, wav)
        pw, _ = sliding_windows(wav, vad=None)
        dn = abs(len(jw) - len(pw))
        worst_n = max(worst_n, dn)
        dwin = float(np.abs(jw - pw).max()) if len(jw) == len(pw) and len(jw) else float("nan")
        if len(jw) == len(pw) and len(jw):
            worst_win = max(worst_win, dwin)
        js = sess.run(["p_ai"], {"window": jw})[0] if len(jw) else np.array([])
        ps = sess.run(["p_ai"], {"window": pw})[0] if len(pw) else np.array([])
        ds = float(np.abs(js - ps).max()) if len(js) == len(ps) and len(js) else float("nan")
        if not np.isnan(ds):
            worst_score = max(worst_score, ds)
        print(f"  {Path(f).name[:34]:34s} windows java {len(jw):2d} / py {len(pw):2d}  "
              f"|Δwin| {dwin:.2e}  mean P(AI) java {js.mean() if len(js) else float('nan'):.4f} "
              f"py {ps.mean() if len(ps) else float('nan'):.4f}  |Δscore| {ds:.2e}")
    print(f"\nworst over {len(files)} files: window-count Δ {worst_n}, "
          f"samples |Δ| {worst_win:.2e}, P(AI) |Δ| {worst_score:.2e}",
          "-> OK" if worst_n == 0 and worst_win < 1e-5 and worst_score < 1e-4 else "-> MISMATCH")


if __name__ == "__main__":
    main()
