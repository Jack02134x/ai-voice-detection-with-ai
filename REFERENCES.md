# References

Everything this project is built from: the languages it is written in, the libraries and tools it
runs on, the AI models inside it, and every dataset it trains or is tested on, online and local.
Versions are the ones installed in `~/ai-things` on 2026-10-11.

- [Languages](#languages)
- [Libraries and tools](#libraries-and-tools)
- [AI models](#ai-models)
- [Data: online](#data-online)
- [Data: local recordings](#data-local-recordings)
- [Licensing notes](#licensing-notes)
- [Citations](#citations)

---

## Languages

### Programming

| Language | Version | Used for |
|---|---|---|
| Python | 3.14.8 | everything except the YouTube downloader (`detector/`, `train.py`, `evaluate.py`, `detect.py`, `build_local_csv.py`, `data_collection/prepare_audio.py`) |
| Bash | 5.3 | `data_collection/youtubethings.sh` (download → clip → delete pipeline) |

### Spoken (in the data)

| Language | Human speech from | AI speech from |
|---|---|---|
| English | all online datasets; local YouTubers (beastboy, BlueSR, fireborn, immy, …) | all online datasets; ChatGPT, Gemini, Neuro-sama, AI YouTube videos, English manhwa recaps |
| Hindi | local only: Hindi manhwa recap channels, Scary Pumpkin podcasts | local only: Sabzi Story Tv, Scary Pumpkin animated stories |
| Arabic, Chinese, German, Romanian, Russian, Spanish | XMAD (Common Voice) | XMAD |
| German, French, Italian, Chinese | CVoiceFake_small (Common Voice) | CVoiceFake_small |
| German, Spanish | ODSS (German only) | ODSS |

Hindi depends entirely on the local collection: no online dataset has it. The other non-English
languages have both classes inside the same dataset (XMAD is balanced per language), so language
can't act as a shortcut. Keep both classes represented in every language you add.

---

## Libraries and tools

| Name | Version | License | Used for |
|---|---|---|---|
| [PyTorch](https://pytorch.org) | 2.14.0 (CUDA 13.0) | BSD-3-Clause | the AASIST model, training, inference |
| [torchaudio](https://github.com/pytorch/audio) | 2.11.0 | BSD-2-Clause | resampling, saving clips, GPU augmentation |
| [torchcodec](https://github.com/pytorch/torchcodec) | 0.16.0 | BSD-3-Clause | decoding audio/video files a chunk at a time (`prepare_audio.py`, `detect.py`); MP3/Opus/Vorbis round trips for codec augmentation (`detector/augment.py`) |
| [NumPy](https://numpy.org) | 2.5.3 | BSD-3-Clause | audio arrays, windows, metrics |
| [pandas](https://pandas.pydata.org) | 3.0.5 | BSD-3-Clause | the local data index (`data/local_data.csv`) |
| [PyArrow](https://arrow.apache.org) | 25.0.1 | Apache-2.0 | reading the online parquet shards row group by row group |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | 1.31.0 | Apache-2.0 | streaming the online datasets over HTTP (`HfFileSystem`) |
| [fsspec](https://github.com/fsspec/filesystem_spec) | 2026.6.0 | BSD-3-Clause | file access layer under `HfFileSystem` |
| [python-soundfile](https://github.com/bastibe/python-soundfile) | 0.14.0 | BSD-3-Clause | reading FLAC/WAV clips |
| [FFmpeg](https://ffmpeg.org) | n9.0.2 | LGPL-2.1+ / GPL | fallback decoder for unusual formats |
| [yt-dlp](https://github.com/yt-dlp/yt-dlp) | 2026.08.19 | Unlicense | downloading YouTube audio (`youtubethings.sh`) |
| [Deno](https://deno.com) | 2.9.7 | MIT | JavaScript runtime yt-dlp uses for YouTube |

Hardware used: NVIDIA RTX 4050 Laptop GPU (6 GB), driver 615.78.08.

---

## AI models

### Inside the detector

| Model | Role | Params | Files | Source | License |
|---|---|---|---|---|---|
| **AASIST** (Jung et al., 2021) | the human-vs-AI classifier: raw waveform → sinc filters → residual encoder → spectral + temporal graph attention → 2-way softmax | 297,866 | `detector/aasist.py` (original code, unmodified), `models/pretrained/AASIST.pth` (official weights, trained on ASVspoof 2019 LA), `models/checkpoints/*.pt` (our fine-tuned weights) | [clovaai/aasist](https://github.com/clovaai/aasist) | MIT (`detector/LICENSE_AASIST`) |
| **AASIST-L** | lighter variant, selectable with `--arch AASIST-L` | 85,306 | `models/pretrained/AASIST-L.pth` | [clovaai/aasist](https://github.com/clovaai/aasist) | MIT |
| **Silero VAD** v5 | speech detector: marks which 32 ms frames contain speech, so silence and background noise get **NO SPEECH** instead of a human/AI verdict, and are kept out of training | ~2 MB | `models/vad/silero_vad.jit` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | MIT (`models/vad/LICENSE_SILERO`) |

Considered but not used: RawNet3. It is a speaker-verification model with no
spoof-detection weights, whereas AASIST ships weights trained for exactly this task.

### Built with

This project was designed and written with **Claude** (Anthropic), working through
**Claude Code** with the **Claude Opus 5.5** model (`claude-opus-5-5`), October 2026. That covers
the code in this folder, the data pipeline, the choice of datasets and models, and this document.

---

## Data: online

Streamed from the Hugging Face organisation
[SpeechAntiSpoofingBenchmarks](https://huggingface.co/SpeechAntiSpoofingBenchmarks), read into
memory with HTTP range requests and **never saved to disk**. Mostly English (see Languages above),
16 kHz mono, labelled bonafide (human) or spoof (AI). Packaged and maintained by Kirill Borodin
(SpeechAntiSpoofingBenchmarks); the original authors are cited below.

### Training

| Dataset | Share of batch* | Clips (human / AI) | AI side | Paper | License |
|---|---|---|---|---|---|
| [ASVspoof5](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof5) | 23.7% | 680,774 (138,688 / 542,086) | 2024 challenge: modern TTS and voice conversion, codecs, crowdsourced speakers | Wang et al., 2024 | ODC-By |
| [InTheWild](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/InTheWild) | 11.1% | 31,779 (19,963 / 11,816) | real-world deepfakes of celebrities and politicians | Müller et al., 2022 | Apache-2.0 |
| [ASVspoof2021_DF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF) | 7.4% | 611,829 (22,617 / 589,212) | 100+ vocoders, MP3/AAC compression | Yamagishi et al., 2021 | other (see card) |
| [CD-ADD](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD) | 7.4% | 20,786 (3,661 / 17,125) | zero-shot TTS: VALL-E, OpenVoice, YourTTS, WhisperSpeech, Seamless | Li et al., 2024 | CC-BY-4.0 |
| [XMAD](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD) | 7.4% | 368,085 (160,656 / 207,429) | XMAD-Bench: 7 languages on Common Voice speakers; kNN-VC, FreeVC, VITS+kNN, XTTS-v2 | arXiv 2506.00462, 2025 | CC-BY-NC-SA-4.0 |
| [DFADD](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/DFADD) | 5.2% | 3,755 (755 / 3,000) | diffusion / flow-matching TTS (Grad-TTS, Matcha-TTS, …) | Du et al., 2024 | MIT |
| [ASVspoof2021_LA](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_LA) | 4.4% | 181,566 (18,452 / 163,114) | 2019 attacks through telephony / transmission channels | Yamagishi et al., 2021 | ODC-By |
| [LibriSeVoc](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/LibriSeVoc) | 3.0% | 18,487 (2,641 / 15,846) | LibriTTS re-synthesised by 6 vocoders | Sun et al., 2023 | CC-BY-SA-4.0 |
| [CVoiceFake_small](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CVoiceFake_small) | 3.0% | 138,136 (23,544 / 114,592) | Common Voice (en/de/fr/it/zh) re-synthesised by vocoders (Griffin-Lim, WORLD, MelGAN, Parallel WaveGAN, …) | Li et al., CCS 2024 (SafeEar) | CC-BY-4.0 |
| [EmoSpoofTTS](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/EmoSpoofTTS) | 3.0% | 36,000 (0 / 36,000) | emotional speech from F5-TTS, CosyVoice, StyleTTS2 (ESD speakers) | Mahapatra et al., 2025 | CC-BY-4.0 |
| [ODSS](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ODSS) | 2.2% | 26,954 (7,961 / 18,993) | VITS and FastPitch+HiFi-GAN in English, German, Spanish | ODSS, IEEE 2023 | CC-BY-SA-4.0 |

\* With default settings; the remaining 22.2% comes from local recordings. Change with
`train.py --weight NAME=W`, preview with `train.py --show-mix`.

XMAD, CVoiceFake_small, EmoSpoofTTS and ODSS were added for v3 (2026-10-11), after v2 scored only
8-16% on SONAR's flow-matching systems (VoiceBox, FlashSpeech, NaturalSpeech 3). **EmoSpoofTTS is
the one exception to "both classes in every source":** its human counterpart (ESD) is licensed
separately and isn't on the Hub. It is kept at a small weight because F5-TTS and CosyVoice are the
closest thing available to those systems. If the model starts flagging calm or emotional read
speech as AI, drop it with `--weight EmoSpoofTTS=0`.

### Validation (during training, never trained on)

- ASVspoof5: last 10 of 200 parquet shards.
- InTheWild: 1 in 5 speakers (by name hash), so validation speakers never appear in training.
- Local and Local+mic: 600 held-out local clips, clean and passed through a simulated room +
  speaker/mic + MP3/Opus chain (`detector/augment.py`, fixed seed). The decision threshold is
  calibrated on the clean Local set.

### Test only (never trained on)

| Dataset | Clips (human / AI) | Content | Paper | License |
|---|---|---|---|---|
| [SONAR](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/SONAR) | 3,948 (2,274 / 1,674) | 8 recent speech-synthesis systems | Li et al., 2024 | CC-BY-NC-4.0 |
| [DeepVoice](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/DeepVoice) | 5,053 (628 / 4,425) | RVC voice conversion between public figures | Bird & Lotfi, 2023 | MIT |

Looked at but not used: ASVspoof2019_LA (its attacks are inside ASVspoof2021_LA), ASVspoof2021_PA
(replay attacks), DECRO (its English human side is ASVspoof2019), HABLA (Spanish only), the
Chinese / Arabic / Japanese-only sets (CFAD, ADD 2022/2023, ArAD, PyAra, J-SPAW), and folder-based
datasets (UncovAI/*, garystafford) that stream one file per HTTP request.

---

## Data: local recordings

Kept in `audio/human/<folder>/` and `audio/ai/<folder>/` as 4.04 s, 16 kHz mono WAV clips,
indexed by `build_local_csv.py` into `data/local_data.csv`. Clips without speech (Silero VAD) are
left out automatically. These recordings were collected for personal research; they are not
redistributable.

### From the earlier SIH project

Counts are clips on disk on 2026-10-11, before the speech check removes no-speech clips.

| Folder | Clips | Origin |
|---|---|---|
| human/beastboy | 569 | YouTube commentary videos (6 recordings), e.g. "What's Up With These GTA 6 Leaks?" |
| human/BlueSR | 551 | YouTube: "I Hosted a Hollow Knight Gameshow (ft @fireb0rn …)" (2 recordings) |
| human/fireborn | 389 | YouTube: "Oddities in Silksong's Code" (2 recordings) |
| human/immy | 348 | YouTube ASMR / dub videos (7 recordings) |
| human/vishesh | 185 | manual human recording ("ep1") |
| human/yairamr | 16,903 | one of the datasets by Hugging Face user [yairamr](https://huggingface.co/yairamr/datasets) (which one wasn't recorded): a speech-enhancement style set with `clean_speech/`, `noisy_speech/` and `noise_only/`. The 4,733 `noise_only` clips contain no speech: they are dropped from training as "human" and used instead as background noise for augmentation (`detector/background.py`) |
| ai/ChatGPT | 559 | ChatGPT voice-mode recordings |
| ai/gemini | 12 | Gemini voice recordings |
| ai/neuro | 48 | Neuro-sama (AI VTuber) clips |
| ai/youtubeslop | 249 | AI-generated YouTube videos (10 videos) |
| ai/Manhwa_explained | 38,475 | 3 English AI manhwa recaps: "(FULL END) When an Engineer Becomes The Greatest Estate Developer in a Fantasy World", "He Awakened LEGENDARY God System & Every SSS-Rank Constellation Was Once His Student!", "[FULL] After DEFEATING the DEMON KING, his greatest CHALLENGE is running a RESTAURANT" (the last one is held out for validation) |

Left out on purpose: `uncovai_tts` (85k single-source AI clips that caused shortcut learning in the
SIH project), HiFi-TTS, CREMA-D, garystafford, `human/silence`.

### From YouTube via `data_collection/youtubethings.sh`

The exact videos are listed in `data_collection/ai_link.txt` and `data_collection/human_link.txt`;
`data_collection/done.txt` records which ones have been processed. At most 1000 clips per video,
one folder per channel. Lists as of 2026-10-11:

| Class | Language | Channel | Videos |
|---|---|---|---|
| AI | English | Frieren Manhwa | 4 |
| AI | English | Manhwa Fresh | 3 |
| AI | English | ManhwaDealer, Novelist Recaps, Gaku Recaps, Tobs Manhwa, Rimuru - Manhwa Recaps | 1 each |
| AI | Hindi | Sabzi Story Tv (animated vegetable stories) | 16 |
| AI | Hindi | Scary Pumpkin (animated horror stories) | 5 |
| Human | Hindi | Queen Manhwa, NV MANHWA EXPLAIN, Vibey Tales, Nur's Manhwa Recap, ManhwaeXp, Modu Manhwa, With Nexus | 1 each |
| Human | Hindi | Scary Pumpkin (multi-host horror podcasts) | 4 |

### Benchmark

`benchmark/ai/<voice>/`, `benchmark/human/<voice>/` and `benchmark/noise/<kind>/` hold hard cases
that are never trained on (e.g. Evil Neuro). `python evaluate.py --folder benchmark` scores them.

---

## Licensing notes

- **SONAR is non-commercial (CC-BY-NC-4.0).** It is only used for testing here; anything
  commercial built on this project should drop it.
- **XMAD is non-commercial and share-alike (CC-BY-NC-SA-4.0)**, and unlike SONAR it is trained on,
  so models trained from v3 on inherit that restriction. For anything commercial, train with
  `--weight XMAD=0`. ODSS and LibriSeVoc are share-alike (CC-BY-SA-4.0).
- **ASVspoof2021_DF** has its own license (`LICENSE.txt` on the dataset card) because its bonafide
  audio comes from the VCC 2018/2020 corpora.
- **yairamr's data:** check the license on the matching dataset card at
  [huggingface.co/yairamr](https://huggingface.co/yairamr/datasets) before sharing anything trained on it.
- **YouTube and other recorded audio** belongs to its creators. It is used here for personal
  research only; don't publish the clips or models trained on them without checking the rights.
- Model code and weights (AASIST, Silero VAD) are MIT and can be redistributed with their license files.

---

## Citations

```bibtex
@article{jung2021aasist,
  author  = {Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
  title   = {{AASIST}: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  journal = {arXiv preprint arXiv:2110.01200},
  year    = {2021}
}

@misc{silero_vad,
  author       = {{Silero Team}},
  title        = {Silero {VAD}: pre-trained enterprise-grade Voice Activity Detector ({VAD}), Number Detector and Language Classifier},
  year         = {2024},
  howpublished = {\url{https://github.com/snakers4/silero-vad}}
}

@inproceedings{wang2024asvspoof5,
  author    = {Wang, Xin and Delgado, H{\'e}ctor and Tak, Hemlata and others},
  title     = {{ASVspoof} 5: Crowdsourced Speech Data, Deepfakes, and Adversarial Attacks at Scale},
  booktitle = {ASVspoof Workshop 2024},
  year      = {2024},
  note      = {arXiv:2408.08739}
}

@inproceedings{yamagishi2021asvspoof,
  author    = {Yamagishi, Junichi and Wang, Xin and Todisco, Massimiliano and Sahidullah, Md and Patino, Jose and others},
  title     = {{ASVspoof} 2021: accelerating progress in spoofed and deepfake speech detection},
  booktitle = {2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
  year      = {2021},
  note      = {arXiv:2109.00537}
}

@article{wang2020asvspoof2019,
  author  = {Wang, Xin and Yamagishi, Junichi and Todisco, Massimiliano and others},
  title   = {{ASVspoof} 2019: A large-scale public database of synthesized, converted and replayed speech},
  journal = {Computer Speech \& Language},
  year    = {2020},
  note    = {arXiv:1911.01601; the pretrained AASIST weights were trained on its LA partition}
}

@inproceedings{muller2022does,
  author    = {M{\"u}ller, Nicolas M. and Czempin, Pavel and Dieckmann, Franziska and Froghyar, Adam and B{\"o}ttinger, Konstantin},
  title     = {Does Audio Deepfake Detection Generalize?},
  booktitle = {Interspeech},
  year      = {2022},
  note      = {arXiv:2203.16263}
}

@article{li2024cdadd,
  author  = {Li, Yuang and Zhang, Min and Ren, Mengxin and Ma, Miaomiao and Wei, Daimeng and others},
  title   = {Cross-Domain Audio Deepfake Detection: Dataset and Analysis},
  journal = {arXiv preprint arXiv:2404.04904},
  year    = {2024}
}

@article{xmadbench2025,
  title   = {{XMAD-Bench}: Cross-Domain Multilingual Audio Deepfake Benchmark},
  journal = {arXiv preprint arXiv:2506.00462},
  year    = {2025}
}

@inproceedings{li2024safeear,
  title     = {{SafeEar}: Content Privacy-Preserving Audio Deepfake Detection},
  author    = {Li, Xinfeng and Li, Kai and Zheng, Yifan and Yan, Chen and Ji, Xiaoyu and Xu, Wenyuan},
  booktitle = {Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security (CCS)},
  year      = {2024},
  doi       = {10.1145/3658644.3670285},
  note      = {source of CVoiceFake}
}

@article{mahapatra2025emotion,
  title   = {Can Emotion Fool Anti-spoofing?},
  author  = {Mahapatra, Aurosweta and Ulgen, Ismail Rasim and Naini, Abinay Reddy and Busso, Carlos and Sisman, Berrak},
  journal = {arXiv preprint arXiv:2505.23962},
  year    = {2025},
  note    = {EmoSpoofTTS}
}

@inproceedings{odss2023,
  title     = {An Open Dataset of Synthetic Speech},
  booktitle = {Proc. IEEE Workshop},
  year      = {2023},
  doi       = {10.5281/zenodo.8370669},
  note      = {ODSS, https://ieeexplore.ieee.org/document/10374863/}
}

@article{du2024dfadd,
  author  = {Du, Jiawei and others},
  title   = {{DFADD}: The Diffusion and Flow-Matching Based Audio Deepfake Dataset},
  journal = {arXiv preprint arXiv:2409.08731},
  year    = {2024}
}

@inproceedings{sun2023librisevoc,
  author    = {Sun, Chengzhe and Jia, Shan and Hou, Shuwei and Lyu, Siwei},
  title     = {{AI}-Synthesized Voice Detection Using Neural Vocoder Artifacts},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  year      = {2023},
  note      = {arXiv:2304.13085}
}

@article{li2024sonar,
  author  = {Li, Xiang and Chen, Pin-Yu and Wei, Wenqi},
  title   = {Where are we in audio deepfake detection? A systematic analysis over generative and detection models},
  journal = {arXiv preprint arXiv:2410.04324},
  year    = {2024}
}

@article{bird2023deepvoice,
  author  = {Bird, Jordan J. and Lotfi, Ahmad},
  title   = {Real-time Detection of {AI}-Generated Speech for DeepFake Voice Conversion},
  journal = {arXiv preprint arXiv:2308.12734},
  year    = {2023}
}
```
