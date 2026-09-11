import sys
import time
import threading

import numpy as np
import sounddevice as sd
import torch

from real_time_test import (
    AIVoiceDetector,
    TARGET_SAMPLE_RATE,
    CLIP_SAMPLES,
    prepare_clip,
    MODEL_PATH,
)


# How often to make a new prediction.
# Each prediction still uses the newest full 4-second window.
PREDICTION_INTERVAL = 1.0
WINDOW_SECONDS = 4.0

# Keep enough audio for the current 4-second window.
buffer_lock = threading.Lock()
audio_buffer = np.zeros(0, dtype=np.float32)

running = True
total_buffered_samples = 0


def load_model(device):
    model = AIVoiceDetector(freeze_encoder=True)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict_clip(model, clip, device):
    clip = prepare_clip(torch.from_numpy(clip))
    clip = clip.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(clip)
        probabilities = torch.softmax(logits, dim=1)[0]

    human = probabilities[0].item()
    ai = probabilities[1].item()

    return human, ai


def audio_callback(indata, frames, callback_time, status):
    global audio_buffer, total_buffered_samples

    if status:
        pass

    samples = indata[:, 0].copy()

    with buffer_lock:
        audio_buffer = np.concatenate((audio_buffer, samples))

        # Keep only the newest 4 seconds.
        if len(audio_buffer) > CLIP_SAMPLES:
            audio_buffer = audio_buffer[-CLIP_SAMPLES:]

        total_buffered_samples += len(samples)


def get_latest_window():
    with buffer_lock:
        if len(audio_buffer) < CLIP_SAMPLES:
            return None

        return audio_buffer.copy()


def clear_screen():
    print("\033[2J\033[H", end="")


def progress_bar(seconds):
    filled = int(seconds / WINDOW_SECONDS * 30)
    filled = max(0, min(30, filled))
    return "[" + "█" * filled + "░" * (30 - filled) + "]"


def draw_screen(
    mic_name,
    device,
    elapsed,
    human,
    ai,
    average_human,
    average_ai,
    prediction_count,
    processing=False,
):
    clear_screen()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                 AI VOICE DETECTOR — MIC TEST                ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║ Microphone : {str(mic_name)[:45]:<45} ║")
    print(f"║ Device     : {str(device)[:45]:<45} ║")
    print("║ Mode       : LIVE MICROPHONE                                ║")
    print("║ Window     : 4.0 seconds                                    ║")
    print("║ Prediction : every 1.0 second (overlapping)                 ║")
    print("╠══════════════════════════════════════════════════════════════╣")

    if elapsed < WINDOW_SECONDS:
        progress = progress_bar(elapsed)
        print(f"║ Listening  {progress} {elapsed:4.1f}/4.0s             ║")
        print("║                                                              ║")
        print("║ Waiting for the first complete 4-second window...           ║")
    else:
        progress = progress_bar(WINDOW_SECONDS)
        state = "ANALYZING" if processing else "LISTENING"
        print(f"║ {state:<10} {progress}  4.0/4.0s             ║")
        print("║                                                              ║")
        print(f"║ Current window                                             ║")
        print(f"║   Human : {human * 100:6.2f}%                                ║")
        print(f"║   AI    : {ai * 100:6.2f}%                                ║")
        print("║                                                              ║")
        print(f"║ Running average ({prediction_count:3d} windows)             ║")
        print(f"║   Human : {average_human * 100:6.2f}%                                ║")
        print(f"║   AI    : {average_ai * 100:6.2f}%                                ║")

    print("╠══════════════════════════════════════════════════════════════╣")
    print("║ Ctrl+C to stop                                               ║")
    print("╚══════════════════════════════════════════════════════════════╝")


def main():
    global running

    if len(sys.argv) != 1:
        print("Usage:")
        print("  python src/mic_test.py")
        print()
        print("This program listens directly to your microphone.")
        return

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    mic_info = sd.query_devices(kind="input")
    mic_name = mic_info["name"]

    print("Loading model...")
    model, checkpoint = load_model(device)

    print(f"Microphone: {mic_name}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    time.sleep(1)

    human = 0.5
    ai = 0.5

    average_human = 0.0
    average_ai = 0.0
    prediction_count = 0

    start_time = time.monotonic()
    next_prediction_at = start_time + WINDOW_SECONDS

    stream = sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
        blocksize=1024,
    )

    try:
        with stream:
            while running:
                now = time.monotonic()
                elapsed = now - start_time

                if now >= next_prediction_at:
                    clip = get_latest_window()

                    if clip is not None:
                        draw_screen(
                            mic_name,
                            device,
                            WINDOW_SECONDS,
                            human,
                            ai,
                            average_human,
                            average_ai,
                            prediction_count,
                            processing=True,
                        )

                        human, ai = predict_clip(
                            model,
                            clip,
                            device,
                        )

                        prediction_count += 1

                        # Running average across all completed windows.
                        average_human += (
                            human - average_human
                        ) / prediction_count

                        average_ai += (
                            ai - average_ai
                        ) / prediction_count

                    # Move exactly one second forward.
                    next_prediction_at += PREDICTION_INTERVAL

                    # If inference took longer than the interval,
                    # skip missed prediction times rather than trying
                    # to process stale windows back-to-back.
                    while next_prediction_at <= time.monotonic():
                        next_prediction_at += PREDICTION_INTERVAL

                draw_screen(
                    mic_name,
                    device,
                    min(elapsed, WINDOW_SECONDS),
                    human,
                    ai,
                    average_human,
                    average_ai,
                    prediction_count,
                    processing=False,
                )

                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        running = False
        sd.stop()

    print()
    print("Stopped microphone test.")

    if prediction_count:
        print()
        print("Final running average:")
        print(f"  Human: {average_human * 100:.2f}%")
        print(f"  AI:    {average_ai * 100:.2f}%")

        result = (
            "AI-generated"
            if average_ai >= average_human
            else "Human"
        )

        print()
        print(f"Result: {result}")


if __name__ == "__main__":
    main()
c