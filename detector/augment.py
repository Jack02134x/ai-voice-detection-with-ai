"""Channel augmentation: what happens to a voice between its source and the model.

A live microphone, and every clip that was re-uploaded or sent through a chat
app, adds things the clean training data mostly lacks. They happen to human
and AI voices alike, so they are applied regardless of the label (like the
backgrounds in background.py):

  room     reverb from the walls (synthetic room impulse response)
  device   the frequency response of a speaker / microphone, sometimes overdriven
  codec    lossy compression: MP3, Opus (YouTube, Discord, WhatsApp), Ogg Vorbis

Order per training clip: room -> background (background.py) -> device -> codec,
i.e. sound in a room, picked up by a mic, then compressed. See CHANNEL_PROB in
config.py for how often each step happens.
"""

import numpy as np
import torch

from .audio_utils import normalize
from .config import CHANNEL_PROB, SAMPLE_RATE

CODECS = [  # (format, bit rates to choose from in bit/s or None, share)
    ("mp3", [24_000, 32_000, 48_000, 64_000, 96_000], 0.40),
    ("opus", [12_000, 16_000, 24_000, 32_000, 48_000], 0.45),
    ("ogg", [None], 0.15),
]


def _convolve(x, h):
    n = 1 << (len(x) + len(h) - 2).bit_length()
    return np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)[:len(x)]


def room_impulse(rng, sr=SAMPLE_RATE):
    """Direct sound + exponentially decaying diffuse tail with a random RT60
    (0.15-0.9 s: small studio to big hall) and direct-to-reverberant ratio."""
    rt60 = rng.uniform(0.15, 0.9)
    t = np.arange(int(sr * rt60)) / sr
    tail = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)        # -60 dB at rt60
    tail = np.convolve(tail, np.ones(3) / 3, mode="same")                # walls absorb highs a bit
    tail[:int(sr * rng.uniform(0.001, 0.01))] = 0                        # first reflection delay
    drr_db = rng.uniform(-3, 12)
    tail *= 10 ** (-drr_db / 20) / (np.sqrt(np.sum(tail ** 2)) + 1e-9)
    tail[0] = 1.0                                                        # direct path
    return tail


def device_response(n, rng, sr=SAMPLE_RATE):
    """Magnitude response of a cheap speaker / mic chain: band-pass with random
    corners plus a few resonances and dips."""
    f = np.maximum(np.fft.rfftfreq(n, 1 / sr), 1.0)
    order = rng.choice([1, 2, 4])
    hp, lp = rng.uniform(60, 400), rng.uniform(3500, 8000)
    mag = 1 / np.sqrt(1 + (hp / f) ** (2 * order)) / np.sqrt(1 + (f / lp) ** (2 * order))
    gain_db = np.zeros_like(f)
    for _ in range(rng.integers(1, 5)):
        fc, g, width = np.exp(rng.uniform(np.log(200), np.log(7000))), rng.uniform(-8, 8), rng.uniform(0.2, 1.0)
        gain_db += g * np.exp(-0.5 * (np.log2(f / fc) / width) ** 2)    # bump `width` octaves wide
    return mag * 10 ** (gain_db / 20)


def codec_roundtrip(x, fmt, bit_rate, sr=SAMPLE_RATE):
    """Encode to `fmt` and decode back, same length as x."""
    from torchcodec.decoders import AudioDecoder
    from torchcodec.encoders import AudioEncoder
    data = AudioEncoder(torch.from_numpy(x)[None], sample_rate=sr).to_tensor(fmt, bit_rate=bit_rate)
    y = AudioDecoder(data, sample_rate=sr, num_channels=1).get_all_samples().data[0].numpy()
    return np.pad(y, (0, max(0, len(x) - len(y))))[:len(x)]


class Channel:
    """`room(clip)` before the background is mixed in, `transmit(clip)` after.
    `prob` = {"room": p, "device": p, "codec": p}; `scale` multiplies them all."""

    def __init__(self, seed=0, prob=None, scale=1.0):
        self.rng = np.random.default_rng(seed)
        self.p = {k: min(1.0, v * scale) for k, v in (prob or CHANNEL_PROB).items()}

    def room(self, clip):
        if self.rng.random() >= self.p["room"]:
            return clip
        return normalize(_convolve(clip, room_impulse(self.rng)))

    def device(self, clip):
        y = np.fft.irfft(np.fft.rfft(clip) * device_response(len(clip), self.rng), len(clip))
        if self.rng.random() < 0.3:                                      # overdriven small speaker
            drive = self.rng.uniform(2, 8)
            y = np.tanh(drive * normalize(y)) / np.tanh(drive)
        return normalize(y)

    def codec(self, clip):
        i = self.rng.choice(len(CODECS), p=[share for *_, share in CODECS])
        fmt, rates, _ = CODECS[i]
        try:
            return normalize(codec_roundtrip(clip, fmt, rates[self.rng.integers(len(rates))]))
        except Exception:                                                # encoder missing: skip
            return clip

    def transmit(self, clip):
        if self.rng.random() < self.p["device"]:
            clip = self.device(clip)
        if self.rng.random() < self.p["codec"]:
            clip = self.codec(clip)
        return clip

    def full(self, wav):
        """All three, always: the 'played through a speaker into a laptop mic and
        sent over a call' test used by evaluate.py --channel."""
        return self.codec(self.device(normalize(_convolve(wav, room_impulse(self.rng)))))
