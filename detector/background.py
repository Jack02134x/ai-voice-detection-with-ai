"""Background sounds mixed under training clips.

Without this the model learned "background noise = human", because noisy
audio was almost always on the human side. Mixing the same backgrounds under
human and AI clips alike removes that shortcut: a noisy AI voice is still AI.

Three kinds, see BG_KINDS in config.py:
  recorded   real noise recordings (yairamr noise_only + anything in noise/)
  babble     2-4 other human voices summed, like a crowd or game characters
  synthetic  white / pink (hiss) / brown (rumble) noise
"""

import glob

import numpy as np

from .audio_utils import load_file, normalize
from .config import BG_KINDS, BG_SNR_DB, NB_SAMP, NOISE_DIR, NOISE_GLOBS, ROOT

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}


def noise_files():
    """All recorded-noise files: NOISE_GLOBS plus everything under NOISE_DIR."""
    files = [f for pattern in NOISE_GLOBS for f in glob.glob(str(ROOT / pattern))]
    if NOISE_DIR.is_dir():
        files += [str(f) for f in NOISE_DIR.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(set(files))


def _fit(wav, n, rng):
    """Random n-sample piece of wav (tiled if it is shorter)."""
    if len(wav) < n:
        wav = np.tile(wav, n // max(len(wav), 1) + 1)
    start = rng.integers(0, len(wav) - n + 1)
    return wav[start:start + n]


def _unit(x):
    return (x / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def colored_noise(n, rng):
    kind = rng.choice(["white", "pink", "brown"])
    if kind == "white":
        return _unit(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spectrum = rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))
    spectrum /= np.sqrt(f) if kind == "pink" else f          # 1/f power or 1/f^2 power
    return _unit(np.fft.irfft(spectrum, n))


class Backgrounds:
    def __init__(self, noise_paths, babble_paths, seed=0, kinds=None, snr_db=BG_SNR_DB):
        self.noise_paths = list(noise_paths)
        self.babble_paths = list(babble_paths)
        self.rng = np.random.default_rng(seed)
        kinds = dict(kinds or BG_KINDS)
        if not self.noise_paths:
            kinds.pop("recorded", None)
        if len(self.babble_paths) < 4:
            kinds.pop("babble", None)
        self.kinds = list(kinds)
        self.kind_p = np.array([kinds[k] for k in self.kinds]) / sum(kinds.values())
        self.snr_db = snr_db

    def sample(self, n=NB_SAMP):
        """One background of n samples at unit RMS, and which kind it is."""
        kind = self.rng.choice(self.kinds, p=self.kind_p)
        try:
            if kind == "recorded":
                path = self.noise_paths[self.rng.integers(len(self.noise_paths))]
                return _unit(_fit(load_file(path), n, self.rng)), kind
            if kind == "babble":
                voices = self.rng.choice(len(self.babble_paths), self.rng.integers(2, 5), replace=False)
                mix = sum(_unit(_fit(load_file(self.babble_paths[i]), n, self.rng)) for i in voices)
                return _unit(mix), kind
        except Exception:
            pass                                   # unreadable file: fall back to synthetic
        return colored_noise(n, self.rng), "synthetic"

    def mix(self, clip, snr_db=None):
        """clip + a background at snr_db (random in BG_SNR_DB if None), re-normalised."""
        snr = self.rng.uniform(*self.snr_db) if snr_db is None else snr_db
        bg, _ = self.sample(len(clip))
        voice_rms = np.sqrt(np.mean(clip ** 2)) + 1e-9
        return normalize(clip + bg * voice_rms * 10 ** (-snr / 20))
