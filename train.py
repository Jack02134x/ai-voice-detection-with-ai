"""Fine-tune AASIST on audio streamed from the Hugging Face Hub + your recordings.

    python train.py                       # fine-tune pretrained AASIST
    python train.py --resume              # continue from models/checkpoints/last.pt
    python train.py --arch AASIST-L       # lighter model

Experiments with a different data mix (each run gets its own folder):

    python train.py --run local40 --init models/checkpoints/best_copy_step11000.pt --steps 4000 \\
        --weight Local=0.4 --folder-weight ai/Manhwa_explained=2

    --weight NAME=W          share of a source (ASVspoof5, CD-ADD, ..., Local); 0 drops it
    --folder-weight F=M      multiply one of your audio/ folders' sampling weight by M
    --show-mix               just print the resulting mix and exit

Ctrl+C at any time saves last.pt so you can resume later (--resume, same --run).
"""

import argparse
import csv
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from torch.utils.data import DataLoader

from detector.config import (ARCHS, BATCH_SIZE, BG_PROB, BG_SNR_DB, CHANNEL_PROB, CHECKPOINT_DIR,
                             GRAD_ACCUM, SAMPLE_RATE, TRAIN_SOURCES)
from detector.data import (LOCAL, TrainStream, default_mix, folder_weights, load_validation_set,
                           local_rows, validation_sources)
from detector.metrics import compute_eer
from detector.model import build_model, load_pretrained


def class_weights(mix):
    """Inverse-frequency CE weights from the expected human share of the mix."""
    human_frac = {name: frac for name, (_, _, frac, _) in TRAIN_SOURCES.items()}
    human_frac[LOCAL] = 0.5                         # local data is sampled 50/50
    total = sum(mix.values())
    human = sum(w * human_frac[name] for name, w in mix.items()) / total
    return torch.tensor([1 - human, human]) * 2    # [human, AI]


def parse_pairs(pairs, valid=None, what="name"):
    out = {}
    for item in pairs:
        key, sep, value = item.rpartition("=")
        if not sep or not key:
            raise SystemExit(f"expected {what}=number, got {item!r}")
        if valid is not None and key not in valid:
            raise SystemExit(f"unknown {what} {key!r}; choose from: {', '.join(sorted(valid))}")
        out[key] = float(value)
    return out


def print_mix(mix, folder_mult, rows):
    total = sum(mix.values())
    print("Data mix (share of every batch):")
    for name, w in sorted(mix.items(), key=lambda kv: -kv[1]):
        print(f"  {name:18s} {w / total:6.1%}")
    if mix.get(LOCAL, 0) > 0 and rows:
        print(f"  inside {LOCAL} (human / AI each 50%):")
        for label, folders in sorted(folder_weights(rows, folder_mult).items()):
            s = sum(folders.values())
            for f, w in sorted(folders.items(), key=lambda kv: -kv[1]):
                print(f"    {f:28s} {w / s:6.1%} of {'AI' if label else 'human'}")


def augment(x):
    """Label-independent augmentation on the GPU batch [B, T]."""
    b, dev = x.size(0), x.device
    # loudness +-6 dB
    x = x * 10 ** (torch.empty(b, 1, device=dev).uniform_(-6, 6) / 20)
    # telephone-style band limiting (8 kHz round trip) on ~15 %
    m = torch.rand(b, device=dev) < 0.15
    if m.any():
        low = AF.resample(x[m], SAMPLE_RATE, 8000)
        x[m] = AF.resample(low, 8000, SAMPLE_RATE)[:, :x.size(1)]
    # background noise at 15-40 dB SNR on ~30 %
    m = (torch.rand(b, 1, device=dev) < 0.3).float()
    snr = torch.empty(b, 1, device=dev).uniform_(15, 40)
    power = x.pow(2).mean(1, keepdim=True)
    x = x + m * torch.randn_like(x) * torch.sqrt(power / 10 ** (snr / 10))
    return x.clamp(-1, 1)


@torch.inference_mode()
def validate(model, val_set, device):
    """Per-source EERs, the pooled EER (used to pick the best checkpoint) and the
    decision threshold. The threshold is the EER point of your own recordings
    when they are in the validation set -- that is the audio the detector is
    used on; the pool is ~80 % ASVspoof5 AI clips and puts it too high."""
    model.eval()
    results, thresholds, all_labels, all_scores = {}, {}, [], []
    for name, (wavs, labels) in val_set.items():
        scores = []
        for i in range(0, len(wavs), 16):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, logits = model(wavs[i:i + 16].to(device))
            scores.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
        scores = torch.cat(scores).numpy()
        results[name], thresholds[name] = compute_eer(labels.numpy(), scores)
        all_labels.append(labels.numpy())
        all_scores.append(scores)
    pooled_eer, threshold = compute_eer(np.concatenate(all_labels), np.concatenate(all_scores))
    model.train()
    return results, pooled_eer, thresholds.get(LOCAL, threshold)


def save(path, model, opt, sched, step, best, threshold, arch, val_sources, mix=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "step": step, "best": best, "threshold": threshold, "arch": arch,
                "val_sources": val_sources, "mix": mix}, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="AASIST", choices=list(ARCHS))
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--accum", type=int, default=GRAD_ACCUM, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--patience", type=int, default=4,
                   help="stop after this many evaluations without a new best (0 = never)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--val-size", type=int, default=600, help="clips per validation source")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--from-scratch", action="store_true", help="don't start from pretrained weights")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run", default="", help="experiment name: checkpoints go to models/checkpoints/<run>/")
    p.add_argument("--init", default=None, help="start from this checkpoint's weights (fresh optimizer)")
    p.add_argument("--weight", action="append", default=[], metavar="NAME=W",
                   help="data-mix weight of a source, repeatable")
    p.add_argument("--folder-weight", action="append", default=[], metavar="FOLDER=M",
                   help="multiplier for one audio/ folder, repeatable")
    p.add_argument("--bg-prob", type=float, default=BG_PROB,
                   help="share of clips that get a background sound mixed in (0 = off)")
    p.add_argument("--channel-aug", type=float, default=1.0,
                   help="multiplier for the room / mic / codec augmentation rates in config.py (0 = off)")
    p.add_argument("--show-mix", action="store_true", help="print the data mix and exit")
    args = p.parse_args()

    mix = default_mix()
    mix.update(parse_pairs(args.weight, valid=set(mix), what="source"))
    rows = local_rows("train")
    folder_mult = parse_pairs(args.folder_weight, valid={r[2] for r in rows}, what="folder")
    print_mix(mix, folder_mult, rows)
    if args.show_mix:
        return

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = CHECKPOINT_DIR / args.run if args.run else CHECKPOINT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    print(f"Checkpoints: {run_dir}")

    model = build_model(args.arch)
    if args.init:
        init = torch.load(args.init, map_location="cpu", weights_only=False)
        args.arch = init["arch"]
        model = build_model(args.arch)
        model.load_state_dict(init["model"])
        print(f"Initialised from {args.init} (step {init.get('step')})")
    elif not args.from_scratch:
        load_pretrained(model, args.arch)
    model.to(device).train()
    print(f"{args.arch}: {sum(p.numel() for p in model.parameters()):,} parameters on {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warmup) * (
        0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(s, args.steps) / args.steps))))

    val_sources = validation_sources()
    if not rows:
        mix.pop(LOCAL, None)
    print(f"Your recordings: {f'used ({len(rows)} train files)' if mix.get(LOCAL, 0) > 0 else 'not used'}")

    step, best, threshold, stale = 0, float("inf"), 0.5, 0
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        step, best, threshold = ckpt["step"], ckpt["best"], ckpt["threshold"]
        print(f"Resumed from step {step} (best val EER {best:.2%})")
        if ckpt.get("val_sources", ["ASVspoof5", "InTheWild"]) != val_sources:
            # A different validation set gives EERs that can't be compared with the old best.
            best = float("inf")
            print(f"Validation set is now {val_sources}: best EER reset")

    print("Loading validation set (streamed into RAM)...")
    val_set = load_validation_set(args.val_size, val_sources)
    eers, pooled, thr = validate(model, val_set, device)
    print(f"[step {step}] val EER {pooled:.2%}  " + "  ".join(f"{k} {v:.2%}" for k, v in eers.items()))

    stream = TrainStream(seed=args.seed + step, mix=mix, folder_mult=folder_mult, bg_prob=args.bg_prob,
                         channel_scale=args.channel_aug)
    if args.bg_prob > 0:
        print(f"Backgrounds: {args.bg_prob:.0%} of clips, {BG_SNR_DB[0]:.0f}-{BG_SNR_DB[1]:.0f} dB, "
              f"{len(stream.noise_paths)} recorded-noise files + babble + synthetic noise")
    if args.channel_aug > 0:
        print("Channel: " + ", ".join(f"{k} {min(1, v * args.channel_aug):.0%}" for k, v in CHANNEL_PROB.items())
              + " of clips (reverb / speaker-mic response / MP3-Opus-Vorbis)")
    loader = DataLoader(stream, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4)
    weights = class_weights(mix).to(device)
    print(f"Class weights [human, AI]: {weights.tolist()}")

    log = open(run_dir / "train_log.csv", "a", newline="")
    writer = csv.writer(log)
    t0, run_loss, run_acc, n, micro = time.time(), 0.0, 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    try:
        for x, y in loader:
            x, y = augment(x.to(device, non_blocking=True)), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                _, logits = model(x, Freq_aug=True)
            loss = F.cross_entropy(logits.float(), y, weight=weights)
            (loss / args.accum).backward()
            run_loss += loss.item()
            run_acc += (logits.argmax(1) == y).float().mean().item()
            n += 1
            micro += 1
            if micro % args.accum:
                continue
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % 50 == 0:
                rate = n * args.batch_size / (time.time() - t0)
                print(f"step {step:6d}  loss {run_loss / n:.4f}  acc {run_acc / n:.2%}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {rate:.0f} clips/s")
                t0, run_loss, run_acc, n = time.time(), 0.0, 0.0, 0

            if step % args.eval_every == 0 or step >= args.steps:
                eers, pooled, thr = validate(model, val_set, device)
                print(f"[step {step}] val EER {pooled:.2%}  " +
                      "  ".join(f"{k} {v:.2%}" for k, v in eers.items()))
                writer.writerow([step, pooled, *eers.values()])
                log.flush()
                if pooled < best:
                    best, threshold, stale = pooled, thr, 0
                    save(best_path, model, opt, sched, step, best, threshold, args.arch, val_sources, mix)
                    print(f"  -> new best, saved {best_path.name} (threshold P(AI) >= {threshold:.3f})")
                else:
                    stale += 1
                    print(f"  (no improvement for {stale} evaluation(s), best {best:.2%})")
                save(last_path, model, opt, sched, step, best, threshold, args.arch, val_sources, mix)
                if args.patience and stale >= args.patience:
                    print("Validation EER stopped improving -- stopping early.")
                    break
            if step >= args.steps:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception:
        # e.g. the network went down for good: keep the progress so --resume can pick it up
        save(last_path, model, opt, sched, step, best, threshold, args.arch, val_sources, mix)
        print(f"\nCrashed at step {step}; saved {last_path}. Fix the cause, then add --resume.")
        raise
    save(last_path, model, opt, sched, step, best, threshold, args.arch, val_sources, mix)
    print(f"Saved {last_path} at step {step}. Best val EER {best:.2%}.")


if __name__ == "__main__":
    main()
