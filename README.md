# AI voice detector (AASIST)

Tells whether a voice recording is a real human or AI-generated (TTS / voice conversion).

See **[REFERENCES.md](REFERENCES.md)** for every language, library, AI model and dataset (online and local) used, with versions, licenses and citations.

- **Model:** AASIST (Jung et al., ICASSP 2022), ~297K parameters, about 15 ms per 4 s window on an RTX 4050.
  Starts from the official ASVspoof2019-LA weights in `models/pretrained/` and is fine-tuned from there.
- **Classifier:** AASIST's own graph-attention back-end (spectral and temporal graphs plus a master node).
  It reads out a 160-d embedding, then one linear layer with a 2-way softmax gives P(human) and P(AI).
  Training uses class-weighted cross-entropy. The decision threshold on P(AI) is calibrated at the
  equal-error-rate point of your held-out recordings (Local validation). Checkpoints up to v2 used
  the pooled validation set instead, which put it too high (0.653); override it with `--threshold`.
  The default `models/checkpoints/best.pt` is v3 (step 18000) with the threshold set by hand to 0.35:
  its Local EER point (0.197) called half of all human voices AI once they went through a mic.
  0.35 balances clean, noisy, mic and SONAR (`threshold_trained` in the file keeps the original).
- **Standard input format:** 16 kHz, mono, float32, leading/trailing silence trimmed, cut into
  64,600-sample (~4.04 s) windows, each RMS-normalised to -26 dBFS. Shorter clips are tiled to
  length. Long files are scanned with 50% overlap, near-silent windows are skipped, and the
  per-window P(AI) values are averaged.
- **Speech detector first:** Silero VAD (MIT, ~2 MB, `models/vad/`) marks which 32 ms frames contain
  speech. Windows with less than 30% speech are skipped, and a recording with under 0.5 s of speech
  gets the verdict **NO SPEECH** instead of human/AI, so silence and background noise are never forced
  into one of the two classes. `build_local_csv.py` uses the same check to keep no-speech clips out of training.
- **Background augmentation:** 40% of training clips, human and AI alike, get a background mixed in at
  0-30 dB: real noise recordings (`audio/human/yairamr/noise_only*` + anything in `noise/`), babble of
  2-4 other voices, or synthetic hiss/rumble. Without it the model learned "background noise = human".
  Put fan/room/game audio or music *without vocals* in `noise/` to add more. `train.py --bg-prob 0` turns it off.
- **Channel augmentation** (`detector/augment.py`), also on human and AI alike:
  - a phone/laptop loudspeaker (no bass, peaky mids) followed by a room on 15%;
  - room reverb on 35% of clips (synthetic room impulse responses);
  - a random speaker/mic frequency response, sometimes overdriven, on 30%;
  - MP3, Opus or Vorbis compression on 25%.

  This is for live microphone use. Reverb was v2's blind spot: a simulated room alone took its EER
  from 3% to 34%. `train.py --channel-aug 0` turns it off; `evaluate.py --channel` runs any test
  through all three effects.
- **Data:** streamed from the Hugging Face Hub (`SpeechAntiSpoofingBenchmarks/*`) and never saved
  to disk. Every training source but EmoSpoofTTS has both human and AI speech, so the model can't
  learn "which dataset is this".
  - Train: ASVspoof5, ASVspoof2021 DF + LA, In-the-Wild (80% of speakers), CD-ADD, DFADD, LibriSeVoc,
    XMAD, CVoiceFake_small, EmoSpoofTTS, ODSS, DeepVoice (RVC voice conversion; 5 of its 8 people)
  - Validation: ASVspoof5 (last 10 shards), In-the-Wild (held-out 20% of speakers), your held-out
    recordings clean (Local) and through the simulated room/mic/codec (Local+mic)
  - Never trained on (generalisation tests): SONAR, and DeepVoice's Obama / Taylor / Margot

- **Your own recordings** (`audio/human/<folder>`, `audio/ai/<folder>`) are indexed by
  `build_local_csv.py` into `data/local_data.csv` and make up ~22% of every batch (`LOCAL_WEIGHT` = 0.30, relative to the other sources).
  Human and AI are sampled 50/50, and folders are weighted by sqrt(size) so small ones still count.
  Whole videos are held out for validation; single-recording folders hold out their last 20%.

## Desktop app (live microphone)

```bash
python app.py                          # Live microphone tab + File analysis tab
python app.py --simulate clip.wav      # feed a file into the live tab instead of the mic
```

Live mode keeps the last ~4 s of microphone audio and scores it every 0.5 s. The speech detector
runs first, so silence shows "LISTENING… (no speech)" instead of a guess. The verdict is the mean
P(AI) of the last 4 speech windows, the graph shows P(AI) over the last 30 s, and the slider at
the bottom moves the decision threshold. Speak into the mic, or play an AI voice from a phone held
near it. The model is trained for replayed audio (simulated loudspeaker, room and mic), but it is a
prototype: through a real mic expect roughly 3 in 4 verdicts right, not the ~97% it gets on clean files.

To make it better at exactly that, record real replays of your clips (human and AI alike) and
retrain: `python data_collection/record_replay.py --per-folder 30` plays clips through the
speakers while recording the mic (~70 min, can be stopped and resumed) into `audio/<class>/replay/`.

## Layout

```
app.py                               desktop app: live microphone + file analysis
detect.py  train.py  evaluate.py      the scripts you run
build_local_csv.py                   indexes audio/ for training
data_collection/   youtubethings.sh + ai_link.txt / human_link.txt -> downloads, clips, deletes;
                   prepare_audio.py cuts any file/folder into clips (max 1000 per file);
                   record_replay.py records real speaker -> room -> mic versions of clips
detector/          library code (config.py has every setting; aasist.py is the original model)
models/pretrained/   official AASIST weights
models/vad/          Silero VAD speech detector
models/checkpoints/  best.pt, last.pt, train_log.csv
data/              local_data.csv (index of audio/), local_sources.csv (per-folder rules)
audio/human/, audio/ai/   your recordings as 4 s clips (trained on)
benchmark/ai/<voice>/, benchmark/human/<voice>/   hard cases, NEVER trained on
benchmark/noise/<kind>/   silence / background noise, should come out as NO SPEECH
.hf_cache/         Hugging Face metadata (no audio is ever saved)
```

## Commands (venv: `~/ai-things`)

```bash
source ~/ai-things/bin/activate

python detect.py recording.wav --timeline     # any format ffmpeg reads

data_collection/youtubethings.sh --dry-run     # list channel/title/length of every link
data_collection/youtubethings.sh               # download + clip + delete (skips videos in done.txt)
python data_collection/prepare_audio.py ~/Downloads/manhwa/* --to ai/Manhwa_explained   # local files
python build_local_csv.py      # re-run whenever you add/remove files in audio/
python train.py                # fine-tune (Ctrl+C saves models/checkpoints/last.pt)
python train.py --resume       # continue later; stops by itself after 4 evals without improvement

python evaluate.py --source Local --limit 600   # your held-out recordings, per folder
python evaluate.py --folder benchmark           # your hard cases, per voice + worst misses
python evaluate.py --source Local --limit 600 --noise-snr 10   # same, with background noise mixed in
python evaluate.py --source Local --limit 600 --channel        # same, through room + mic + MP3/Opus
python evaluate.py --source Local --limit 600 --channel replay # same, phone speaker -> room -> laptop mic
python detect.py recording.wav --threshold 0.5                 # override the stored decision threshold

# experiments with a different data mix (own folder, never touches models/checkpoints/best.pt)
python train.py --show-mix --weight Local=0.5 --folder-weight human/immy=2      # preview
python train.py --run local50 --init models/checkpoints/best_copy_step11000.pt --steps 4000 --weight Local=0.5
python evaluate.py --source Local --limit 600 --checkpoint models/checkpoints/local50/best.pt

python evaluate.py --source SONAR             # unseen modern TTS
python evaluate.py --source InTheWild --limit 2000
```

`detect.py` / `evaluate.py` use `models/checkpoints/best.pt` once it exists, and the pretrained weights before that.
