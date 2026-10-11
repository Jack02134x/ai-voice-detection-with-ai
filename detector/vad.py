"""Is anyone speaking? Silero VAD (MIT, models/vad/silero_vad.jit).

The AASIST classifier only knows "human" or "AI", so on silence or background
noise (fans, traffic, keyboard, room hum) it still has to pick one. This
small neural voice-activity detector runs first and gives a speech
probability for every 32 ms of audio; windows without enough speech are
never sent to the classifier.
"""

import warnings

import numpy as np
import torch

from .config import SAMPLE_RATE, VAD_PATH, VAD_THRESHOLD

FRAME = 512   # samples per VAD frame at 16 kHz (32 ms)


class SpeechDetector:
    def __init__(self, path=VAD_PATH, threshold=VAD_THRESHOLD):
        with warnings.catch_warnings():   # torch.jit.load warns on Python 3.14 but works
            warnings.simplefilter("ignore", FutureWarning)
            self.model = torch.jit.load(str(path), map_location="cpu").eval()
        self.threshold = threshold

    @torch.inference_mode()
    def frame_probs(self, wav: np.ndarray) -> np.ndarray:
        """16 kHz mono audio -> speech probability per 32 ms frame."""
        if len(wav) < FRAME:
            wav = np.pad(wav, (0, FRAME - len(wav)))
        x = torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)).unsqueeze(0)
        # Silero runs tiny 32 ms steps: many CPU threads only add overhead
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            self.model.reset_states()
            return self.model.audio_forward(x, SAMPLE_RATE)[0].numpy()
        finally:
            torch.set_num_threads(threads)

    def speech_mask(self, wav: np.ndarray) -> np.ndarray:
        """Boolean per 32 ms frame: True where someone is speaking."""
        return self.frame_probs(wav) >= self.threshold

    def speech_ratio(self, wav: np.ndarray) -> float:
        return float(self.speech_mask(wav).mean())
