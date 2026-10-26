"""Generate AI-labelled training speech with open-source TTS (guaranteed labels, many voices).

    tts/.venv/bin/python tts/generate.py                 # Kokoro + Piper, default amounts
    tts/.venv/bin/python tts/generate.py --engine kokoro --per-voice 20
    tts/.venv/bin/python tts/generate.py --dry-run       # just show the plan

Why: v5 called NotebookLM human 92% of the time. The AI side of the training data came from a
handful of YouTube recap channels sharing one or two TTS voices, so the model learned "that
narrator = AI" instead of "AI in general". Many different synthetic voices from two different
TTS families fix that cheaply.

Output (then run `python build_local_csv.py` and train):
  audio/ai/tts_kokoro/kokoro_<voice>_<nnnn>.wav    Kokoro (StyleTTS2-style), US/UK English + Hindi
  audio/ai/tts_piper/piper_<voice>_<nnnn>.wav      Piper (VITS), US/UK English + Hindi
Each voice is one recording group, so build_local_csv.py holds whole voices out for
validation. Files are saved at the engine's native rate; the training pipeline resamples
to 16 kHz exactly like every other clip. Existing files are skipped, so it can be resumed.

Only English and Hindi voices are used: those are the languages on the human side, so the
model can't learn "other language = AI". Runs on CPU in its own venv (tts/.venv).
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "audio" / "ai"
KOKORO_DIR = HERE / "models" / "kokoro"
PIPER_DIR = HERE / "models" / "piper"

# Kokoro voice prefixes we use -> language code. (a = US English, b = UK English, h = Hindi)
KOKORO_LANGS = {"af": "en-us", "am": "en-us", "bf": "en-gb", "bm": "en-gb", "hf": "hi", "hm": "hi"}
# Piper multi-speaker models: how many of their speakers to use as separate voices
PIPER_MULTI_SPEAKERS = 8


def load_texts(name):
    lines = [l.strip() for l in (HERE / name).read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def utterance(rng, sentences):
    """1-3 sentences joined (mostly 2), so clips run roughly 3-10 s."""
    k = rng.choices([1, 2, 3], weights=[2, 5, 3])[0]
    return " ".join(rng.sample(sentences, k))


def kokoro_jobs(per_en, per_hi):
    from kokoro_onnx import Kokoro
    k = Kokoro(str(KOKORO_DIR / "kokoro-v1.0.onnx"), str(KOKORO_DIR / "voices-v1.0.bin"))
    voices = sorted(v for v in k.get_voices() if v[:2] in KOKORO_LANGS)
    jobs = []
    for v in voices:
        lang = KOKORO_LANGS[v[:2]]
        jobs.append((f"kokoro_{v}", lang, per_hi if lang == "hi" else per_en, v))

    def synth(voice, text, speed, lang):
        audio, sr = k.create(text, voice=voice, speed=speed, lang=lang)
        return np.asarray(audio, np.float32), sr
    return jobs, synth


def piper_jobs(per_en, per_hi, rng):
    from piper import PiperVoice, SynthesisConfig
    jobs, cache = [], {}
    for onnx in sorted(PIPER_DIR.glob("*.onnx")):
        name = onnx.name.removesuffix(".onnx")             # e.g. en_US-amy-medium
        lang = "hi" if name.startswith("hi_") else "en"
        voice = PiperVoice.load(str(onnx))
        cache[name] = voice
        n_spk = getattr(voice.config, "num_speakers", 1) or 1
        per = per_hi if lang == "hi" else per_en
        short = name.split("-")[1]                         # amy, vctk, pratham, ...
        if n_spk > 1:
            for spk in sorted(rng.sample(range(n_spk), min(PIPER_MULTI_SPEAKERS, n_spk))):
                jobs.append((f"piper_{short}_s{spk}", lang, max(per // 2, 10), (name, spk)))
        else:
            jobs.append((f"piper_{short}", lang, per, (name, None)))

    def synth(key, text, speed, lang):
        name, spk = key
        cfg = SynthesisConfig(speaker_id=spk, length_scale=1.0 / speed)
        chunks = list(cache[name].synthesize(text, syn_config=cfg))
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        return audio, chunks[0].sample_rate
    return jobs, synth


def run(engine, jobs, synth, texts, out_dir, rng, dry):
    out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(n for _, _, n, _ in jobs)
    print(f"\n{engine}: {len(jobs)} voices, {total} clips -> {out_dir.relative_to(ROOT)}", flush=True)
    if dry:
        for group, lang, n, _ in jobs:
            print(f"  {group:28s} {lang:6s} {n} clips")
        return 0
    made = 0
    for gi, (group, lang, n, key) in enumerate(jobs, 1):
        sentences = texts["hi" if lang == "hi" else "en"]
        vrng = random.Random(f"{group}")                    # same text plan on every resume
        for i in range(1, n + 1):
            path = out_dir / f"{group}_{i:04d}.wav"
            text, speed = utterance(vrng, sentences), vrng.uniform(0.85, 1.2)
            if path.exists():
                continue
            try:
                audio, sr = synth(key, text, speed, lang)
                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio / peak * 0.9
                sf.write(path, audio, sr, subtype="PCM_16")
                made += 1
            except Exception as e:
                print(f"  ! {group} #{i}: {type(e).__name__}: {e}", flush=True)
        print(f"  [{gi}/{len(jobs)}] {group:28s} done", flush=True)
    return made


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", choices=["kokoro", "piper", "all"], default="all")
    p.add_argument("--per-voice", type=int, default=45, help="clips per English voice")
    p.add_argument("--per-voice-hi", type=int, default=70, help="clips per Hindi voice (fewer voices)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    rng = random.Random(args.seed)
    texts = {"en": load_texts("texts_en.txt"), "hi": load_texts("texts_hi.txt")}
    print(f"texts: {len(texts['en'])} English, {len(texts['hi'])} Hindi sentences")

    made = 0
    if args.engine in ("kokoro", "all"):
        jobs, synth = kokoro_jobs(args.per_voice, args.per_voice_hi)
        made += run("Kokoro", jobs, synth, texts, OUT / "tts_kokoro", rng, args.dry_run)
    if args.engine in ("piper", "all"):
        if not any(PIPER_DIR.glob("*.onnx")):
            sys.exit("no Piper voices in tts/models/piper (see tts/README.md)")
        jobs, synth = piper_jobs(args.per_voice, args.per_voice_hi, rng)
        made += run("Piper", jobs, synth, texts, OUT / "tts_piper", rng, args.dry_run)
    if not args.dry_run:
        print(f"\nmade {made} new clips. Next: python build_local_csv.py, then train.")


if __name__ == "__main__":
    main()
