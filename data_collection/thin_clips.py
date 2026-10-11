"""Keep at most N clips per recording, spread evenly over its length; delete the rest.

    python data_collection/thin_clips.py audio/ai/Manhwa_explained --keep 1000 --dry-run
    python data_collection/thin_clips.py audio/ai/Manhwa_explained --keep 1000

A recording is a set of clips named <name>_<number>.wav in one folder (what
prepare_audio.py makes). The number is the clip's position in the original
video, so keeping evenly spaced numbers keeps the whole video's variety, the
same way prepare_audio.py --max-clips does. Searches the folder recursively.
Afterwards run: python build_local_csv.py
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

CLIP = re.compile(r"^(.*)_(\d+)\.wav$")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--keep", type=int, required=True, help="clips to keep per recording")
    p.add_argument("--dry-run", action="store_true", help="only show what would be deleted")
    args = p.parse_args()

    root = Path(args.folder).expanduser()
    if not root.is_dir():
        raise SystemExit(f"Not a folder: {root}")

    recordings = defaultdict(list)            # (folder, name) -> [(number, path)]
    for f in root.rglob("*.wav"):
        m = CLIP.match(f.name)
        if m:
            recordings[(f.parent, m.group(1))].append((int(m.group(2)), f))

    total_kept = total_deleted = 0
    for (folder, name), clips in sorted(recordings.items()):
        clips.sort()
        if len(clips) <= args.keep:
            keep = set(range(len(clips)))
        else:
            keep = set(np.linspace(0, len(clips) - 1, args.keep).round().astype(int).tolist())
        delete = [f for i, (_, f) in enumerate(clips) if i not in keep]
        print(f"{name[:70]}: {len(clips)} clips -> keep {len(clips) - len(delete)}, delete {len(delete)}")
        if not args.dry_run:
            for f in delete:
                f.unlink()
        total_kept += len(clips) - len(delete)
        total_deleted += len(delete)

    verb = "Would delete" if args.dry_run else "Deleted"
    print(f"\n{len(recordings)} recording(s): keeping {total_kept} clips. {verb} {total_deleted}.")
    if args.dry_run:
        print("Nothing was deleted (dry run).")
    else:
        print("Now run: python build_local_csv.py")


if __name__ == "__main__":
    main()
