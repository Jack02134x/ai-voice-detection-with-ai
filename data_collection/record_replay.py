"""Record real "played through a speaker, picked up by a mic" versions of your clips.

The live demo plays AI voices from a phone into the laptop mic. Simulated rooms and
speakers (detector/augment.py) only get so close; this records the real thing. It
plays clips from audio/ through the speakers and records the microphone at the
same time, for human AND AI clips alike, so "sounds replayed" never means "AI".

    python data_collection/record_replay.py --per-folder 30            # ~25 min for every folder
    python data_collection/record_replay.py --per-folder 5 --dry-run   # what would be played
    python data_collection/record_replay.py --list-devices
    python data_collection/record_replay.py --output-device 5 --input-device 1

Output: audio/<class>/replay/<original folder>/<same file name>.wav. build_local_csv.py
gives each one the train/val split of the clip it came from. Already-recorded clips are
skipped, so it can be stopped (Ctrl+C) and continued any time. Afterwards:
python build_local_csv.py, then train.

Tips: keep the room quiet, set the volume to a normal listening level, and vary it
between sessions (laptop speakers, a phone over Bluetooth, a speaker across the room).
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402
import sounddevice as sd                             # noqa: E402
import soundfile as sf                               # noqa: E402

from detector.audio_utils import load_file, normalize  # noqa: E402
from detector.config import LOCAL_AUDIO_ROOT, LOCAL_CSV, SAMPLE_RATE  # noqa: E402

AUDIO_ROOT = LOCAL_AUDIO_ROOT / "audio"
TAIL = 0.6        # seconds recorded after the clip ends (speaker / room latency + echo)


def align(recorded, played):
    """Crop the recording to the played clip, compensating the playback latency."""
    n = len(played)
    search = min(len(recorded) - n, int(0.5 * SAMPLE_RATE))
    if search <= 0:
        return recorded[:n]
    spec = np.fft.rfft(recorded[:n + search], 2 * (n + search))
    ref = np.fft.rfft(played, 2 * (n + search))
    corr = np.fft.irfft(spec * np.conj(ref))[:search + 1]
    lag = int(np.argmax(corr))
    return recorded[lag:lag + n]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-folder", type=int, default=30, help="clips per audio/ folder")
    p.add_argument("--classes", nargs="+", default=["human", "ai"])
    p.add_argument("--input-device", type=int, default=None)
    p.add_argument("--output-device", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--list-devices", action="store_true")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    df = pd.read_csv(LOCAL_CSV)
    df = df[~df["folder"].str.endswith("/replay")]
    rng = random.Random(args.seed)
    todo = []
    for (folder, label), g in df.groupby(["folder", "label"]):
        cls = folder.split("/")[0]
        if cls not in args.classes:
            continue
        files = sorted(g["file"])
        for f in rng.sample(files, min(args.per_folder, len(files))):
            out = AUDIO_ROOT / cls / "replay" / folder.split("/", 1)[1] / Path(f).name
            if not out.exists():
                todo.append((f, out))
    rng.shuffle(todo)                       # mix human and AI so volume / room drift hits both
    minutes = len(todo) * (4.04 + TAIL + 0.3) / 60
    print(f"{len(todo)} clips to record (~{minutes:.0f} min). Output: {AUDIO_ROOT}/<class>/replay/")
    if args.dry_run:
        for f, out in todo[:20]:
            print(f"  {Path(f).parent.name}/{Path(f).name}")
        return

    sd.default.device = (args.input_device, args.output_device)
    print(f"Playing on: {sd.query_devices(kind='output')['name']}\n"
          f"Recording : {sd.query_devices(kind='input')['name']}\nStarting in 3 s (Ctrl+C to stop)...")
    time.sleep(3)
    done = 0
    try:
        for f, out in todo:
            clip = normalize(load_file(f)) * 0.7
            played = np.concatenate([clip, np.zeros(int(TAIL * SAMPLE_RATE), np.float32)])
            rec = sd.playrec(played, samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocking=True)[:, 0]
            got = align(rec, clip)
            level = 20 * np.log10(np.sqrt(np.mean(got ** 2)) + 1e-9)
            if level < -55:
                print(f"  ! {out.name}: mic level {level:.0f} dBFS -- is the volume up / the right mic?")
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, got, SAMPLE_RATE)
            done += 1
            print(f"  [{done}/{len(todo)}] {out.parent.parent.parent.name}/{out.parent.name}/{out.name}  "
                  f"mic {level:.0f} dBFS")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"Recorded {done} clips. Now run: python build_local_csv.py")


if __name__ == "__main__":
    main()
