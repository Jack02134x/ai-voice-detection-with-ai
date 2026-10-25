"""Export the detector to ONNX for the Android app, and verify it matches PyTorch.

    python export_onnx.py                       # uses models/checkpoints/best.pt
    python export_onnx.py --checkpoint models/checkpoints/v5/best.pt

Writes to android/app/src/main/assets/:
  aasist.onnx     one 4.04 s window [1, 64600] float32 -> P(AI) scalar
  model.json      threshold, sample rate, window length, step name (the app reads this)

AASIST is tiny (~300K params), so the ONNX file is ~1-2 MB. The app runs it with ONNX
Runtime; preprocessing (resample, trim, 4 s windows at -26 dBFS, averaging) is ported to
Java in VoiceCheck/, and audio_check.py here proves the Java matches this Python.
"""

import argparse
import json
from pathlib import Path

import detector.config  # noqa: F401
import numpy as np
import torch

from detector.audio_utils import eval_clip, load_file
from detector.config import NB_SAMP, SAMPLE_RATE
from detector.model import Detector, build_model

ASSETS = detector.config.ROOT / "android" / "assets"


class AasistProb(torch.nn.Module):
    """Wraps AASIST so ONNX takes [N, 64600] and returns P(AI) [N]."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        _, logits = self.model(x)
        return torch.softmax(logits, dim=1)[:, 1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()

    det = Detector(args.checkpoint, device="cpu", use_vad=False)
    wrapper = AasistProb(det.model).eval()
    ASSETS.mkdir(parents=True, exist_ok=True)
    onnx_path = ASSETS / "aasist.onnx"

    dummy = torch.randn(1, NB_SAMP)
    torch.onnx.export(
        wrapper, (dummy,), str(onnx_path),
        input_names=["window"], output_names=["p_ai"],
        dynamic_axes={"window": {0: "batch"}, "p_ai": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False)
    print(f"wrote {onnx_path}  ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    model_json = {"threshold": det.threshold, "sample_rate": SAMPLE_RATE, "window": NB_SAMP,
                  "target_dbfs": detector.config.TARGET_DBFS, "source": det.source}
    (ASSETS / "model.json").write_text(json.dumps(model_json, indent=2))
    print(f"wrote {ASSETS / 'model.json'}: {model_json}")

    # ---- verify: ONNX vs PyTorch on real clips ----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    import pandas as pd
    df = pd.read_csv(detector.config.LOCAL_CSV)
    files = (df[df.folder == "human/vishesh"].file.tolist()[:4]
             + df[df.folder.str.startswith("ai/")].file.tolist()[:4])
    worst = 0.0
    with torch.inference_mode():
        for f in files:
            win = eval_clip(load_file(f))[None]                       # [1, 64600]
            pt = float(wrapper(torch.from_numpy(win))[0])
            on = float(sess.run(["p_ai"], {"window": win})[0][0])
            worst = max(worst, abs(pt - on))
            print(f"  {Path(f).name[:40]:40s} torch {pt:.4f}  onnx {on:.4f}  |Δ| {abs(pt - on):.2e}")
    print(f"max |PyTorch - ONNX| over {len(files)} clips: {worst:.2e}",
          "-> OK" if worst < 1e-4 else "-> TOO LARGE")


if __name__ == "__main__":
    main()
