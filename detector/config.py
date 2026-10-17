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

# Online data is a FIXED pool of 50,000 clips (Jack, 2026-10-13: "i will allow 50k 4 sec audio
# files ai and human included from online sources"). Exactly which rows are used is written to
# data/online_subset.csv the first time (build_online_subset in data.py), so every run trains on
# the same clips. Training reads only those rows, streamed as before (never saved to disk).
#
# name: (human clips, AI clips, shards held out for validation)
TRAIN_SOURCES = {
    "ASVspoof5":   (7000, 6000, 10),  # 2024 challenge: modern TTS/VC, crowd-sourced speakers, codecs
    "InTheWild":   (7000, 5000,  0),  # real-world recordings of public figures (speaker split below)
    "XMAD":        (6000, 6000,  0),  # 7 languages, Common Voice phone/laptop mics; kNN-VC/FreeVC/XTTS-v2
    "CD-ADD":      (3000, 3000,  0),  # zero-shot TTS: VALL-E, OpenVoice, YourTTS, ...
    "DFADD":       ( 700, 2300,  0),  # diffusion / flow-matching TTS
    "EmoSpoofTTS": (   0, 2000,  0),  # emotional F5-TTS / CosyVoice / StyleTTS2 (AI only!)
    "DeepVoice":   ( 400, 1600,  0),  # RVC voice conversion, training speakers only (split below)
}
# Dropped on 2026-10-13 as repetitive (mostly older vocoders saying the same corpora again):
# ASVspoof2021_DF, ASVspoof2021_LA, LibriSeVoc, CVoiceFake_small, ODSS. Still usable as tests.
ONLINE_WEIGHT = 1.0          # share of every batch from the online pool (vs LOCAL_WEIGHT below)

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
LOCAL_WEIGHT = 1.0           # your recordings: with ONLINE_WEIGHT 1.0 that is half of every batch
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
