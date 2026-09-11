import sys
import time

import torch
import torchaudio

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


def progress_bar(progress, width=30):
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


# ─────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample → 16 kHz
    if sample_rate != TARGET_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=TARGET_SAMPLE_RATE,
        )

        waveform = resampler(waveform)

    return waveform.squeeze(0)


def prepare_clip(waveform):
    length = waveform.shape[0]

    # Pad final short clip
    if length < CLIP_SAMPLES:
        padding = CLIP_SAMPLES - length

        waveform = torch.nn.functional.pad(
            waveform,
            (0, padding),
        )

    # Normalize
    waveform = waveform - waveform.mean()

    std = waveform.std()

    if std > 0:
        waveform = waveform / std

    return waveform


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def load_model(device):
    print(f"{CYAN}Loading model...{RESET}")

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

    print(
        f"{GREEN}✓ Model loaded{RESET} "
        f"(epoch {checkpoint['epoch']}, "
        f"val {checkpoint['val_accuracy'] * 100:.2f}%)"
    )

    return model


# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():

    if len(sys.argv) != 2:
        print(
            f"{BOLD}Usage:{RESET}\n"
            f"  python src/predict.py <audio_file>"
        )
        return

    audio_path = sys.argv[1]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    clear_screen()

    print()
    print(
        f"{BOLD}{CYAN}"
        "╭──────────────────────────────────────────────╮"
        "│           AI VOICE DETECTOR                  │"
        "╰──────────────────────────────────────────────╯"
        f"{RESET}"
    )

    print()

    print(f"{WHITE}File:{RESET} {audio_path}")

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(0)

        print(
            f"{WHITE}Device:{RESET} "
            f"{GREEN}CUDA{RESET} ({gpu})"
        )
    else:
        print(
            f"{WHITE}Device:{RESET} "
            f"{YELLOW}CPU{RESET}"
        )

    print()

    # Load model
    model = load_model(device)

    print()

    # Load audio
    print(f"{CYAN}Loading audio...{RESET}")

    waveform = load_audio(audio_path)

    total_samples = waveform.shape[0]
    total_duration = total_samples / TARGET_SAMPLE_RATE

    num_clips = (
        total_samples + CLIP_SAMPLES - 1
    ) // CLIP_SAMPLES

    print(
        f"{GREEN}✓ Audio loaded{RESET}"
    )

    print(
        f"{WHITE}Duration:{RESET} "
        f"{format_time(total_duration)}"
    )

    print(
        f"{WHITE}Windows:{RESET} "
        f"{num_clips}"
    )

    print()

    # Results for every window
    ai_probabilities = []
    human_probabilities = []

    # ─────────────────────────────────────────
    # Process each 4-second window
    # ─────────────────────────────────────────

    for clip_index in range(num_clips):

        start_sample = clip_index * CLIP_SAMPLES
        end_sample = min(
            start_sample + CLIP_SAMPLES,
            total_samples,
        )

        start_time = start_sample / TARGET_SAMPLE_RATE
        end_time = end_sample / TARGET_SAMPLE_RATE

        clip = waveform[
            start_sample:end_sample
        ]

        human_probability, ai_probability = predict_clip(
            model,
            clip,
            device,
        )

        ai_probabilities.append(ai_probability)
        human_probabilities.append(human_probability)

        progress = (clip_index + 1) / num_clips

        clear_screen()

        print()
        print(
            f"{BOLD}{CYAN}"
            "╭──────────────────────────────────────────────╮"
            "│           AI VOICE DETECTOR                  │"
            "╰──────────────────────────────────────────────╯"
            f"{RESET}"
        )

        print()

        print(
            f"{WHITE}File:{RESET} {audio_path}"
        )

        print(
            f"{WHITE}Duration:{RESET} "
            f"{format_time(total_duration)}"
        )

        print()

        print(
            f"{WHITE}Processing:{RESET} "
            f"{progress_bar(progress)} "
            f"{progress * 100:6.2f}%"
        )

        print()

        print(
            f"{WHITE}Window:{RESET} "
            f"{format_time(start_time)} → "
            f"{format_time(end_time)}"
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

        # Small delay makes the UI readable during very fast inference.
        # Remove this if you want maximum speed.
        time.sleep(0.03)

    # ─────────────────────────────────────────
    # Overall result
    # ─────────────────────────────────────────

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

    human_windows = num_clips - ai_windows

    overall_ai = average_ai >= average_human

    clear_screen()

    print()
    print(
        f"{BOLD}{CYAN}"
        "╭──────────────────────────────────────────────╮"
        "│              ANALYSIS COMPLETE               │"
        "╰──────────────────────────────────────────────╯"
        f"{RESET}"
    )

    print()

    print(
        f"{WHITE}File:{RESET} {audio_path}"
    )

    print(
        f"{WHITE}Duration:{RESET} "
        f"{format_time(total_duration)}"
    )

    print(
        f"{WHITE}Windows analyzed:{RESET} "
        f"{num_clips}"
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
        f"{ai_windows}/{num_clips}"
    )

    print(
        f"{WHITE}Human windows:{RESET} "
        f"{human_windows}/{num_clips}"
    )

    print()

    if overall_ai:

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