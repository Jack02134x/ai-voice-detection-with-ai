"""Saving live-microphone sessions (used by server.py and app.py).

Every session is kept whole in data_collection/mic_sessions/ (to check what the mic
delivers); a labelled one is also cut into 4.04 s training clips in
audio/<class>/mic_<name>/<session>_<nnn>.wav. The session timestamp is the clip-name
prefix, so build_local_csv.py groups clips by session for the train/val split.
"""

import time

import numpy as np
import soundfile as sf

from .config import LOCAL_AUDIO_ROOT, NB_SAMP, ROOT, SAMPLE_RATE

SESSIONS_DIR = ROOT / "data_collection" / "mic_sessions"


def clean_name(name):
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name or "")) or "mic"


def save_session(audio, label=None, name="mic"):
    """audio: float32 at 16 kHz; label: None, 'human' or 'ai'. Returns a short summary."""
    audio = np.asarray(audio, np.float32)
    if len(audio) < SAMPLE_RATE:
        return "Session shorter than 1 s, not saved."
    stamp, name = time.strftime("%Y%m%d_%H%M%S"), clean_name(name)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(SESSIONS_DIR / f"{label or 'unlabelled'}_{name}_{stamp}.wav", audio, SAMPLE_RATE)
    if label not in ("human", "ai"):
        return "Session kept in data_collection/mic_sessions/."
    out = LOCAL_AUDIO_ROOT / "audio" / label / f"mic_{name}"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for i in range(0, len(audio) - NB_SAMP + 1, NB_SAMP):
        sf.write(out / f"{stamp}_{n:03d}.wav", audio[i:i + NB_SAMP], SAMPLE_RATE)
        n += 1
    return f"Saved {n} clips as {label.upper()} in audio/{label}/mic_{name}/ (run build_local_csv.py, then train)."
