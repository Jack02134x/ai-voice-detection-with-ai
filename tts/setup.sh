#!/usr/bin/env bash
# Recreates the TTS generator: its own venv (tts/.venv) and the voice models (tts/models/).
# Both are git-ignored (~1.5 GB), so run this once on a fresh clone. Needs espeak-ng installed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

[[ -x .venv/bin/python ]] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q kokoro-onnx piper-tts soundfile

mkdir -p models/kokoro models/piper
for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  [[ -f models/kokoro/$f ]] || curl -fsSL -o "models/kokoro/$f" \
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/$f"
done

.venv/bin/python -m piper.download_voices --download-dir models/piper \
  en_US-amy-medium en_US-bryce-medium en_US-hfc_female-medium en_US-hfc_male-medium en_US-joe-medium \
  en_US-kristin-medium en_US-lessac-medium en_US-ljspeech-medium en_US-norman-medium en_US-ryan-medium \
  en_US-libritts_r-medium \
  en_GB-alan-medium en_GB-cori-medium en_GB-jenny_dioco-medium en_GB-northern_english_male-medium en_GB-vctk-medium \
  hi_IN-pratham-medium hi_IN-priyamvada-medium hi_IN-rohan-medium

echo "ready: tts/.venv/bin/python tts/generate.py"
