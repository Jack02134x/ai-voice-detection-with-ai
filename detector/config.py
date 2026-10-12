"""Shared settings for the AI-voice detector.

Label convention everywhere in this project:  0 = human (bonafide), 1 = AI (spoof).
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # project folder

# Keep every Hugging Face file (stream metadata, locks) inside this folder so
# nothing is written to ~/.cache. Must be set before `datasets` is imported.
os.environ.setdefault("HF_HOME", str(ROOT / ".hf_cache"))

PRETRAINED_DIR = ROOT / "models" / "pretrained"
VAD_PATH = ROOT / "models" / "vad" / "silero_vad.jit"   # speech / no-speech detector
CHECKPOINT_DIR = ROOT / "models" / "checkpoints"
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Standard input format: every waveform is turned into this before the model
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
NB_SAMP = 64_600                 # 4.0375 s -- the window AASIST was designed for
WINDOW_HOP = NB_SAMP // 2        # 50 % overlap when scanning a long file
TARGET_DBFS = -26.0              # RMS loudness each window is normalised to
SILENCE_DB_BELOW_PEAK = 40.0     # frames this far below the loudest are "silence"
MIN_SPEECH_RATIO = 0.30          # windows with less speech than this are skipped
VAD_THRESHOLD = 0.5              # Silero speech probability above which a 32 ms frame is speech
MIN_SPEECH_SECONDS = 0.5         # less speech than this in a whole recording -> "no speech"

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
ARCHS = {
    "AASIST": {           # ~297K params
        "nb_samp": NB_SAMP,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    },
    "AASIST-L": {         # ~85K params, lighter / faster
        "nb_samp": NB_SAMP,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 24], [24, 24]],
        "gat_dims": [24, 32],
        "pool_ratios": [0.4, 0.5, 0.7, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    },
}

# ---------------------------------------------------------------------------
# Data -- streamed from the Hugging Face Hub, never saved to disk.
# Every source contains BOTH human and AI speech recorded under the same
# conditions, so the model can't learn "which dataset is this" as a shortcut.
# ---------------------------------------------------------------------------
HUB_ORG = "SpeechAntiSpoofingBenchmarks"

# name: (sampling weight, number of parquet shards, fraction human, shards held out for validation)
TRAIN_SOURCES = {
    "ASVspoof5":        (0.32, 200, 0.20, 10),  # 2024 challenge, modern TTS/VC + codecs
    "ASVspoof2021_DF":  (0.10,  80, 0.04,  0),  # 100+ vocoders, MP3/AAC compression
    "ASVspoof2021_LA":  (0.06,  24, 0.10,  0),  # telephony codecs / transmission
    "InTheWild":        (0.15,   8, 0.63,  0),  # real-world celebrity deepfakes (speaker split below)
    "CD-ADD":           (0.10,   8, 0.18,  0),  # zero-shot TTS: VALL-E, OpenVoice, YourTTS, ...
    "DFADD":            (0.07,   2, 0.20,  0),  # diffusion / flow-matching TTS
    "LibriSeVoc":       (0.04,   8, 0.14,  0),  # six neural vocoders on LibriTTS
    # added for v3 -- more languages, everyday microphones, newer TTS:
    "XMAD":             (0.10, 200, 0.44,  0),  # 7 languages, Common Voice mics, kNN-VC/FreeVC/XTTS-v2
    "CVoiceFake_small": (0.04,  14, 0.17,  0),  # Common Voice en/de/fr/it/zh through 5 vocoders
    "EmoSpoofTTS":      (0.04,   9, 0.00,  0),  # emotional F5-TTS / CosyVoice / StyleTTS2 (AI only!)
    "ODSS":             (0.03,   8, 0.30,  0),  # VITS / FastPitch, en/de/es
    # added for v4: RVC voice conversion (the usual live voice changer). Speaker split below.
    "DeepVoice":        (0.03,   2, 0.21,  0),
}

# DeepVoice people whose real speech and RVC conversions between each other are
# never trained on (evaluate.py --source DeepVoice tests on exactly these).
DEEPVOICE_HELDOUT = {"obama", "taylor", "margot"}

# In-the-Wild speakers whose name hashes into this bucket are never trained on
# (used for validation/testing instead) -> speaker-disjoint split.
ITW_HELDOUT_BUCKETS = 5   # 1 in 5 speakers held out

# Never used for training: pure generalisation tests.
TEST_SOURCES = ["SONAR"]

# ---------------------------------------------------------------------------
# Your own recordings (YouTube + real-world), read from local disk.
# Layout: <LOCAL_AUDIO_ROOT>/audio/human/<folder>/...  and  .../audio/ai/<folder>/...
# Index them with `python build_local_csv.py` (re-run after adding files).
# ---------------------------------------------------------------------------
LOCAL_AUDIO_ROOT = Path(os.environ.get("LOCAL_AUDIO_ROOT", ROOT))
LOCAL_SOURCES_CSV = DATA_DIR / "local_sources.csv"   # per-folder role / group_regex (from the SIH project)
LOCAL_CSV = Path(os.environ.get("LOCAL_CSV", DATA_DIR / "local_data.csv"))
# folders never used. ai/Scary_Pumpkin was deleted on 2026-10-12 (not confirmed to be AI -- an
# uncertain label does more harm than the data is worth); listed so it is skipped if re-downloaded.
# Its human podcasts (human/Scary_Pumpkin) stay.
LOCAL_EXCLUDE = {"ai/uncovai_tts", "human/silence", "ai/Scary_Pumpkin"}
LOCAL_WEIGHT = 0.30          # share of each training batch drawn from your recordings
LOCAL_VAL_FRACTION = 0.2     # recordings (whole videos / speakers) held out for validation
LOCAL_FOLDER_CAP = 2000      # folders bigger than this are sampled as if they had this many files

# ---------------------------------------------------------------------------
# Background augmentation: mixed into human AND AI training clips alike, so the
# model can't use "there is background noise" as a clue for either class.
# ---------------------------------------------------------------------------
NOISE_DIR = ROOT / "noise"       # drop extra background audio here: fans, rooms, game audio, music WITHOUT vocals
NOISE_GLOBS = ["audio/human/yairamr/noise_only*"]   # existing noise-only recordings (relative to ROOT)
BG_PROB = 0.4                    # share of training clips that get a background
BG_SNR_DB = (0.0, 30.0)          # voice-to-background loudness ratio, drawn uniformly
BG_KINDS = {"recorded": 0.4, "babble": 0.3, "synthetic": 0.3}   # which kind of background

# Channel augmentation (detector/augment.py), also label-independent: share of
# training clips that get room reverb / a speaker-mic frequency response / lossy
# compression. Makes the model ignore the recording chain -- needed for live mic use.
# Room matters most: v2 (trained without it) went from 3 % to 34 % EER on reverberant audio.
# "speaker" = played through a small loudspeaker (phone / laptop) first, then always a room:
# the "AI voice from a phone held up to the mic" case. (v3 used room 0.3, device 0.25, codec 0.3.)
CHANNEL_PROB = {"speaker": 0.15, "room": 0.35, "device": 0.3, "codec": 0.25}

# ~0.24 GB VRAM per clip during training: 12 x 2 accumulation = effective batch 24
# (the original AASIST recipe) in ~3 GB, so it fits a 6 GB laptop GPU.
BATCH_SIZE = 12
GRAD_ACCUM = 2
SHUFFLE_BUFFER = 200
