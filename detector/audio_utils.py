"""Turn any audio into the standard format the model sees:

    16 kHz, mono, float32, leading/trailing silence trimmed,
    cut into 64,600-sample (~4.04 s) windows, each RMS-normalised to -26 dBFS.

The same functions are used for training, evaluation and inference so the
model never sees a different format at test time.
"""

import io
import subprocess

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF

from .config import (MIN_SPEECH_RATIO, MIN_SPEECH_SECONDS, NB_SAMP, SAMPLE_RATE,
                    SILENCE_DB_BELOW_PEAK, TARGET_DBFS, WINDOW_HOP)

FRAME = 400       # 25 ms
HOP = 160         # 10 ms
VAD_FRAME = 512   # Silero VAD frame, 32 ms


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def to_standard_rate(wav: np.ndarray, sr: int) -> np.ndarray:
    """Any (samples,) or (samples, channels) array -> mono float32 at 16 kHz."""
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = AF.resample(torch.from_numpy(wav), sr, SAMPLE_RATE).numpy()
    return wav


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode an in-memory audio file (FLAC/WAV/OGG/MP3) from the stream."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    return to_standard_rate(wav, sr)


def load_file(path: str) -> np.ndarray:
    """Load any audio/video file ffmpeg understands, as 16 kHz mono float32."""
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        return to_standard_rate(wav, sr)
    except Exception:
        pass
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def iter_file_chunks(path, chunk_seconds=600):
    """Yield (start_sec, 16 kHz mono float32 chunk) through a file of any length.

    Decodes with torchcodec a chunk at a time (already resampled and mixed to
    mono), so multi-hour recordings never sit in RAM all at once.
    """
    from torchcodec.decoders import AudioDecoder
    decoder = AudioDecoder(str(path), sample_rate=SAMPLE_RATE, num_channels=1)
    duration = decoder.metadata.duration_seconds
    start = 0.0
    while duration is None or start < duration:
        try:
            chunk = decoder.get_samples_played_in_range(start, start + chunk_seconds).data
        except (RuntimeError, ValueError, IndexError):
            return                                            # past the end
        if chunk.numel() == 0:
            return
        yield start, chunk[0].numpy().astype(np.float32)
        start += chunk_seconds


# ---------------------------------------------------------------------------
# Silence handling
# ---------------------------------------------------------------------------

def frame_db(wav: np.ndarray) -> np.ndarray:
    """Per-frame RMS level in dBFS (25 ms frames, 10 ms hop)."""
    if len(wav) < FRAME:
        wav = np.pad(wav, (0, FRAME - len(wav)))
    frames = np.lib.stride_tricks.sliding_window_view(wav, FRAME)[::HOP]
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def speech_mask(wav: np.ndarray) -> np.ndarray:
    """Boolean per-frame mask: True where the frame is not silence."""
    db = frame_db(wav)
    threshold = max(db.max() - SILENCE_DB_BELOW_PEAK, -60.0)
    return db > threshold


def loud_bounds(wav: np.ndarray):
    """(start, end) sample range without leading/trailing silence (50 ms margin)."""
    mask = speech_mask(wav)
    if not mask.any():
        return 0, len(wav)
    idx = np.flatnonzero(mask)
    return max(idx[0] * HOP - 800, 0), min(idx[-1] * HOP + FRAME + 800, len(wav))


def trim_silence(wav: np.ndarray) -> np.ndarray:
    """Cut leading and trailing silence (keeps a 50 ms margin)."""
    start, end = loud_bounds(wav)
    return wav[start:end]


# ---------------------------------------------------------------------------
# Fixed-length windows
# ---------------------------------------------------------------------------

def normalize(wav: np.ndarray) -> np.ndarray:
    """RMS-normalise to TARGET_DBFS, then make sure nothing clips."""
    rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
    wav = wav * (10 ** (TARGET_DBFS / 20) / rms)
    peak = np.abs(wav).max()
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav.astype(np.float32)


def fit_length(wav: np.ndarray, random_crop: bool) -> np.ndarray:
    """Exactly NB_SAMP samples: crop if long, repeat (tile) if short."""
    n = len(wav)
    if n >= NB_SAMP:
        start = np.random.randint(0, n - NB_SAMP + 1) if random_crop else 0
        return wav[start:start + NB_SAMP]
    reps = NB_SAMP // max(n, 1) + 1
    return np.tile(wav, reps)[:NB_SAMP]


def training_clip(wav: np.ndarray) -> np.ndarray:
    """One standard-format window from a training utterance (random crop)."""
    return normalize(fit_length(trim_silence(wav), random_crop=True))


def eval_clip(wav: np.ndarray) -> np.ndarray:
    """One standard-format window, deterministic (start of the speech)."""
    return normalize(fit_length(trim_silence(wav), random_crop=False))


def sliding_windows(wav: np.ndarray, vad=None):
    """Split a full-length recording into overlapping standard windows.

    Returns (windows [N, NB_SAMP], start_times_sec [N]). A window is kept only
    if at least MIN_SPEECH_RATIO of it is speech -- judged by `vad` (a
    SpeechDetector) when given, otherwise by loudness. If no window qualifies
    but the recording has at least MIN_SPEECH_SECONDS of speech, the window
    with the most speech is kept. Returns empty arrays when there is no speech.
    Recordings shorter than one window are tiled.
    """
    offset, end = loud_bounds(wav)
    wav = wav[offset:end]
    if vad is not None:
        mask, hop = vad.speech_mask(wav), VAD_FRAME
    else:
        mask, hop = speech_mask(wav), HOP

    def speech_seconds(s, n):
        return mask[s // hop:(s + n) // hop].sum() * hop / SAMPLE_RATE

    empty = np.empty((0, NB_SAMP), np.float32), np.empty(0)
    if mask.sum() * hop / SAMPLE_RATE < MIN_SPEECH_SECONDS:
        return empty
    if len(wav) <= NB_SAMP:
        return normalize(fit_length(wav, False))[None], np.array([offset / SAMPLE_RATE])

    starts = list(range(0, len(wav) - NB_SAMP + 1, WINDOW_HOP))
    if starts[-1] + NB_SAMP < len(wav):       # cover the tail too
        starts.append(len(wav) - NB_SAMP)
    speech = [speech_seconds(s, NB_SAMP) for s in starts]
    keep = [s for s, sp in zip(starts, speech) if sp >= MIN_SPEECH_RATIO * NB_SAMP / SAMPLE_RATE]
    if not keep:                              # sparse speech: use the window with the most
        keep = [starts[int(np.argmax(speech))]]

    windows = np.stack([normalize(wav[s:s + NB_SAMP]) for s in keep])
    return windows, (np.array(keep) + offset) / SAMPLE_RATE
