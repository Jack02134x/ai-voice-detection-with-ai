# Synthetic AI speech (open-source TTS)

Generates **AI-labelled** training clips from many synthetic voices, so the detector learns
"AI in general" rather than the one or two narrator voices most YouTube recap channels share.
Labels are guaranteed: every clip here was made by a TTS engine.

```bash
tts/setup.sh                                   # once: venv + models (~1.5 GB, git-ignored)
tts/.venv/bin/python tts/generate.py --dry-run # show the plan
tts/.venv/bin/python tts/generate.py           # generate the full plan (resumable; ~1 h on CPU)
python build_local_csv.py                      # then index and train as usual
```

| Engine | Architecture | Voices used | Clips | License |
|---|---|---|---|---|
| [Kokoro](https://github.com/hexgrad/kokoro) (via kokoro-onnx) | StyleTTS2-style | 32: 20 US + 8 UK English, 4 Hindi | 1,540 (all) | Apache-2.0 |
| [Piper](https://github.com/rhasspy/piper) | VITS | 33: 10 US + 4 UK single, 8 VCTK + 8 LibriTTS-R speakers, 3 Hindi | 384: all 3 Hindi voices × 70, English 1–10 per voice | MIT (code); each voice's license is in its `.onnx.json` model card |

Output goes to `audio/ai/tts_kokoro/` and `audio/ai/tts_piper/`, one file per utterance
(1–3 sentences from `texts_en.txt` / `texts_hi.txt`, random speed 0.85–1.2). Each voice is a
recording group, so `build_local_csv.py` holds whole **voices** out for validation — that
measures generalisation to voices the model never trained on.

Only English and Hindi are used, matching the human side, so the model can't learn
"another language = AI".

**What this does and doesn't fix (measured with v5, 2026-10-26):** v5 already called most of
these AI (Piper English 98%, Kokoro English 89%), so they add *variety* and fix the thin
**Hindi** AI side (Piper Hindi 33%). They do **not** stand in for modern conversational TTS:
v5 called NotebookLM AI only 8% of the time. For that, collect real NotebookLM / Gemini /
ChatGPT-voice / ElevenLabs audio.

**What was actually generated (2026-10-26):** all of Kokoro (1,540) and Piper's Hindi voices
(210), but only a test batch of Piper's English voices (174), since v5 already caught those
98% of the time. Total 1,924 clips. Running `generate.py` again would fill in the rest of
Piper English if it is ever wanted.
