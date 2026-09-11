import sys
import time
import threading

import numpy as np
import torch
import torchaudio
import sounddevice as sd

from model import AIVoiceDetector


MODEL_PATH = "models/ai_voice_detector/best_model.pt"

TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION = 4
CLIP_SAMPLES = TARGET_SAMPLE_RATE * CLIP_DURATION


# ─────────────────────────────────────────────
# Terminal colors
# ─────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"

CYAN = "\033[96m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
WHITE = "\033[97m"
GRAY = "\033[90m"


def clear_screen():
    print("\033[2J\033[H", end="")


def progress_bar(progress, width=32):
    progress = max(0.0, min(1.0, progress))

    filled = int(width * progress)
    empty = width - filled

    return (
        f"{GREEN}"
        + "█" * filled
        + f"{GRAY}"
        + "░" * empty
        + f"{RESET}"
    )


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)

    return f"{minutes:02d}:{seconds:02d}"


def load_audio(path):
    """
    Load audio twice conceptually:
      - original waveform is used for playback
      - analysis waveform is converted to mono / 16 kHz
    """

    waveform, sample_rate = torchaudio.load(path)

    # Keep original audio for playback.
    playback_waveform = waveform

    # Analysis copy: stereo → mono
    analysis_waveform = waveform

    if analysis_waveform.shape[0] > 1:
        analysis_waveform = analysis_waveform.mean(
            dim=0,
            keepdim=True,
        )

    # Analysis copy: resample → 16 kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )

        analysis_waveform = resampler(analysis_waveform)

    analysis_waveform = analysis_waveform.squeeze(0)

    return (
        playback_waveform,
        sample_rate,
        analysis_waveform,
    )


def prepare_clip(waveform):
    length = waveform.shape[0]

    if length < CLIP_SAMPLES:
        padding = CLIP_SAMPLES - length

        waveform = torch.nn.functional.pad(
            waveform,
            (0, padding),
        )

    waveform = waveform - waveform.mean()

    std = waveform.std()

    if std > 0:
        waveform = waveform / std

    return waveform


def load_model(device):
    model = AIVoiceDetector(
        freeze_encoder=True
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict_clip(model, waveform, device):
    waveform = prepare_clip(waveform)

    waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)

    with torch.no_grad():
        logits = model(waveform)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

    human_probability = probabilities[0, 0].item()
    ai_probability = probabilities[0, 1].item()

    return human_probability, ai_probability


def show_header(audio_path, total_duration, device):
    print()
    print(
        f"{BOLD}{CYAN}"
        "╭──────────────────────────────────────────────╮"
        "│          REAL-TIME AI VOICE DETECTOR        │"
        "╰──────────────────────────────────────────────╯"
        f"{RESET}"
    )

    print()

    print(f"{WHITE}File:{RESET} {audio_path}")
    print(
        f"{WHITE}Duration:{RESET} "
        f"{format_time(total_duration)}"
    )

    if device.type == "cuda":
        print(
            f"{WHITE}Device:{RESET} "
            f"{GREEN}CUDA{RESET} "
            f"({torch.cuda.get_device_name(0)})"
        )
    else:
        print(
            f"{WHITE}Device:{RESET} "
            f"{YELLOW}CPU{RESET}"
        )

    print()


def main():
    if len(sys.argv) != 2:
        print(
            f"{BOLD}Usage:{RESET}\n"
            f"  python src/realtime_test.py <audio_file>"
        )
        return

    audio_path = sys.argv[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    clear_screen()

    print(
        f"{CYAN}Loading model...{RESET}"
    )

    model, checkpoint = load_model(device)

    print(
        f"{GREEN}✓ Model loaded{RESET} "
        f"(epoch {checkpoint['epoch']}, "
        f"val {checkpoint['val_accuracy'] * 100:.2f}%)"
    )

    print(
        f"{CYAN}Loading audio...{RESET}"
    )

    (
        playback_waveform,
        playback_sample_rate,
        analysis_waveform,
    ) = load_audio(audio_path)

    total_samples = analysis_waveform.shape[0]
    total_duration = total_samples / TARGET_SAMPLE_RATE

    num_clips = (
        total_samples + CLIP_SAMPLES - 1
    ) // CLIP_SAMPLES

    # Convert playback audio to NumPy for sounddevice.
    # sounddevice expects [samples, channels].
    playback_numpy = (
        playback_waveform
        .transpose(0, 1)
        .contiguous()
        .numpy()
        .astype(np.float32)
    )

    show_header(
        audio_path,
        total_duration,
        device,
    )

    print(
        f"{WHITE}Windows:{RESET} "
        f"{num_clips} × {CLIP_DURATION}s"
    )

    print()
    print(
        f"{BOLD}{GREEN}"
        "▶ PLAYING"
        f"{RESET}"
    )

    print(
        f"{GRAY}"
        "The first prediction appears after the first "
        f"{CLIP_DURATION} seconds."
        f"{RESET}"
    )

    print()

    # Start playback.
    sd.play(
        playback_numpy,
        playback_sample_rate,
        blocking=False,
    )

    start_time = time.monotonic()

    ai_probabilities = []
    human_probabilities = []

    try:
        for clip_index in range(num_clips):

            start_sample = clip_index * CLIP_SAMPLES
            end_sample = min(
                start_sample + CLIP_SAMPLES,
                total_samples,
            )

            window_start = start_sample / TARGET_SAMPLE_RATE
            window_end = end_sample / TARGET_SAMPLE_RATE

            # Wait until this 4-second section has actually played.
            target_time = start_time + window_end
            remaining = target_time - time.monotonic()

            if remaining > 0:
                time.sleep(remaining)

            clip = analysis_waveform[
                start_sample:end_sample
            ]

            human_probability, ai_probability = predict_clip(
                model,
                clip,
                device,
            )

            human_probabilities.append(
                human_probability
            )

            ai_probabilities.append(
                ai_probability
            )

            elapsed = min(
                window_end,
                total_duration,
            )

            progress = elapsed / total_duration

            clear_screen()

            show_header(
                audio_path,
                total_duration,
                device,
            )

            print(
                f"{WHITE}Playback:{RESET} "
                f"{format_time(elapsed)} / "
                f"{format_time(total_duration)}"
            )

            print()

            print(
                f"{WHITE}Progress:{RESET} "
                f"{progress_bar(progress)} "
                f"{progress * 100:6.2f}%"
            )

            print()

            print(
                f"{WHITE}Analyzing window:{RESET} "
                f"{format_time(window_start)} → "
                f"{format_time(window_end)}"
            )

            print()

            print(
                f"{WHITE}Human:{RESET} "
                f"{human_probability * 100:7.2f}%"
            )

            print(
                f"{WHITE}AI:{RESET}    "
                f"{ai_probability * 100:7.2f}%"
            )

            print()

            if ai_probability >= human_probability:
                print(
                    f"{BOLD}{RED}"
                    "● AI-GENERATED"
                    f"{RESET}"
                )
            else:
                print(
                    f"{BOLD}{GREEN}"
                    "● HUMAN"
                    f"{RESET}"
                )

            print()

            # Running average.
            average_ai = sum(ai_probabilities) / len(
                ai_probabilities
            )

            print(
                f"{GRAY}"
                f"Running AI probability: "
                f"{average_ai * 100:.2f}%"
                f"{RESET}"
            )

    except KeyboardInterrupt:
        print()
        print(
            f"{YELLOW}Stopping playback...{RESET}"
        )

    finally:
        sd.stop()

    if not ai_probabilities:
        return

    # Final result.
    average_ai = sum(ai_probabilities) / len(
        ai_probabilities
    )

    average_human = sum(human_probabilities) / len(
        human_probabilities
    )

    ai_windows = sum(
        p >= 0.5
        for p in ai_probabilities
    )

    human_windows = len(ai_probabilities) - ai_windows

    clear_screen()

    show_header(
        audio_path,
        total_duration,
        device,
    )

    print(
        f"{BOLD}{CYAN}"
        "╭──────────────────────────────────────────────╮"
        "│              ANALYSIS COMPLETE              │"
        "╰──────────────────────────────────────────────╯"
        f"{RESET}"
    )

    print()

    print(
        f"{WHITE}Windows analyzed:{RESET} "
        f"{len(ai_probabilities)}"
    )

    print()

    print(
        f"{WHITE}Average Human:{RESET} "
        f"{average_human * 100:7.2f}%"
    )

    print(
        f"{WHITE}Average AI:{RESET}    "
        f"{average_ai * 100:7.2f}%"
    )

    print()

    print(
        f"{WHITE}AI windows:{RESET} "
        f"{ai_windows}/{len(ai_probabilities)}"
    )

    print(
        f"{WHITE}Human windows:{RESET} "
        f"{human_windows}/{len(ai_probabilities)}"
    )

    print()

    if average_ai >= average_human:
        print(
            f"{BOLD}{RED}"
            "████  RESULT: AI-GENERATED  ████"
            f"{RESET}"
        )
    else:
        print(
            f"{BOLD}{GREEN}"
            "████  RESULT: HUMAN  ████"
            f"{RESET}"
        )

    print()


if __name__ == "__main__":
    main()
