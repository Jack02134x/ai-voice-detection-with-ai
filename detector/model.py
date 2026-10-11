"""Model construction, checkpoint loading and the inference-time Detector."""

from pathlib import Path

import numpy as np
import torch

from .aasist import Model
from .audio_utils import iter_file_chunks, sliding_windows
from .config import ARCHS, CHECKPOINT_DIR, PRETRAINED_DIR, SAMPLE_RATE, VAD_PATH


def build_model(arch="AASIST"):
    return Model(ARCHS[arch])


def load_pretrained(model, arch="AASIST"):
    """Load the official ASVspoof2019-LA weights shipped in ./pretrained.

    The original release uses output index 0 = spoof, 1 = bonafide. This
    project uses 0 = human, 1 = AI, so the two rows of the final layer are swapped.
    """
    sd = torch.load(PRETRAINED_DIR / f"{arch}.pth", map_location="cpu", weights_only=True)
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    sd["out_layer.weight"] = sd["out_layer.weight"].flip(0)
    sd["out_layer.bias"] = sd["out_layer.bias"].flip(0)
    model.load_state_dict(sd)
    return model


class Detector:
    """Scores audio already in 16 kHz mono float32 form.

    Uses models/checkpoints/best.pt when it exists, otherwise the pretrained weights.
    A speech detector (Silero VAD) decides first which parts contain speech;
    recordings with no speech get "no_speech": True instead of a human/AI verdict.
    """

    def __init__(self, checkpoint=None, arch="AASIST", device=None, batch_size=16, use_vad=True,
                 threshold=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        # the first layer's output is ~18 MB per window, so keep CPU batches small for RAM
        self.batch_size = batch_size if self.device.type == "cuda" else 4
        checkpoint = Path(checkpoint) if checkpoint else CHECKPOINT_DIR / "best.pt"

        if checkpoint.exists():
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.arch = ckpt["arch"]
            self.model = build_model(self.arch)
            self.model.load_state_dict(ckpt["model"])
            self.threshold = float(ckpt.get("threshold", 0.5))
            self.source = f"{checkpoint.name} (step {ckpt.get('step')})"
        else:
            self.arch = arch
            self.model = load_pretrained(build_model(arch), arch)
            self.threshold = 0.5
            self.source = f"pretrained {arch} (ASVspoof2019-LA, not fine-tuned)"
        if threshold is not None:          # override the one calibrated during training
            self.threshold = float(threshold)

        self.model.to(self.device).eval()

        self.vad = None
        if use_vad and VAD_PATH.exists():
            from .vad import SpeechDetector
            self.vad = SpeechDetector()

    @torch.inference_mode()
    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        """[N, NB_SAMP] standard windows -> P(AI) for each, shape [N]."""
        if len(windows) == 0:
            return np.empty(0, np.float32)
        probs = []
        for i in range(0, len(windows), self.batch_size):
            x = torch.from_numpy(windows[i:i + self.batch_size]).to(self.device)
            _, logits = self.model(x)
            probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        return torch.cat(probs).numpy()

    def _result(self, probs, times, duration):
        no_speech = len(probs) == 0
        ai_prob = float("nan") if no_speech else float(probs.mean())
        return {
            "no_speech": no_speech,
            "ai_probability": ai_prob,
            "is_ai": (not no_speech) and ai_prob >= self.threshold,
            "threshold": self.threshold,
            "window_starts": times,
            "window_probs": probs,
            "duration": duration,
        }

    def score_waveform(self, wav: np.ndarray) -> dict:
        """Full-length recording -> overall P(AI) plus a per-window timeline."""
        windows, times = sliding_windows(wav, self.vad)
        return self._result(self.score_windows(windows), times, len(wav) / SAMPLE_RATE)

    def score_file(self, path, chunk_seconds=600, progress=None) -> dict:
        """Like score_waveform, but reads the file 10 minutes at a time so
        recordings of any length work. progress(seconds_done) is called per chunk."""
        all_probs, all_times, duration = [], [], 0.0
        for start, chunk in iter_file_chunks(path, chunk_seconds):
            windows, times = sliding_windows(chunk, self.vad)
            all_probs.append(self.score_windows(windows))
            all_times.append(times + start)
            duration = start + len(chunk) / SAMPLE_RATE
            if progress:
                progress(duration)
        if duration == 0:
            raise ValueError("no audio decoded")
        return self._result(np.concatenate(all_probs), np.concatenate(all_times), duration)
