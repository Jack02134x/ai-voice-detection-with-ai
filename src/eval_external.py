from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from model import AIVoiceDetector


# ============================================================
# Configuration
# ============================================================

AUDIO_DIR = Path("~/Videos/snip_4sec").expanduser()
MODEL_FILE = "ai_voice_detector.pt"
OUTPUT_FILE = "data/external_results.csv"

TARGET_SAMPLE_RATE = 16_000
CLIP_DURATION = 4
CLIP_SAMPLES = TARGET_SAMPLE_RATE * CLIP_DURATION

BATCH_SIZE = 8


# ============================================================
# Dataset
# ============================================================

class ExternalDataset(Dataset):
    def __init__(self, audio_dir):
        self.files = sorted(
            path
            for path in audio_dir.rglob("*")
            if path.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}
        )

        if not self.files:
            raise RuntimeError(f"No audio files found in {audio_dir}")

    def __len__(self):
        return len(self.files)

    def load_audio(self, path):
        waveform, sample_rate = torchaudio.load(str(path))

        # Stereo -> mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sample_rate != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=TARGET_SAMPLE_RATE,
            )
            waveform = resampler(waveform)

        return waveform.squeeze(0)

    def prepare_clip(self, waveform):
        length = waveform.shape[0]

        # This evaluator expects the files to already be 4-second clips.
        if length > CLIP_SAMPLES:
            waveform = waveform[:CLIP_SAMPLES]

        elif length < CLIP_SAMPLES:
            padding = CLIP_SAMPLES - length
            waveform = torch.nn.functional.pad(
                waveform,
                (0, padding),
            )

        # Same normalization as training/evaluation
        waveform = waveform - waveform.mean()

        std = waveform.std()

        if std > 0:
            waveform = waveform / std

        return waveform

    def __getitem__(self, index):
        path = self.files[index]

        waveform = self.load_audio(path)
        waveform = self.prepare_clip(waveform)

        return waveform, str(path)


# ============================================================
# Main
# ============================================================

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print()
    print(f"Audio directory: {AUDIO_DIR}")

    dataset = ExternalDataset(AUDIO_DIR)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Audio files: {len(dataset)}")

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = AIVoiceDetector(freeze_encoder=True)
    model = model.to(device)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    model.eval()

    results = []

    with torch.inference_mode():

        for waveforms, paths in loader:

            waveforms = waveforms.to(
                device,
                non_blocking=True,
            )

            logits = model(waveforms)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            ai_probabilities = probabilities[:, 1]

            predictions = logits.argmax(dim=1)

            for path, prediction, ai_probability in zip(
                paths,
                predictions,
                ai_probabilities,
            ):

                ai_probability = ai_probability.item()
                prediction = prediction.item()

                results.append(
                    {
                        "file": path,
                        "prediction": prediction,
                        "ai_probability": ai_probability,
                    }
                )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df["prediction_name"] = results_df[
        "prediction"
    ].map(
        {
            0: "HUMAN",
            1: "AI",
        }
    )

    results_df["ai_probability_percent"] = (
        results_df["ai_probability"] * 100
    )

    # Sort by probability, most AI-like first
    results_df = results_df.sort_values(
        "ai_probability",
        ascending=False,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    ai_prob = results_df["ai_probability"]

    print()
    print("=" * 70)
    print("EXTERNAL TEST RESULTS")
    print("=" * 70)

    print(f"Files:              {len(results_df)}")
    print(
        f"Predicted AI:       "
        f"{(results_df['prediction'] == 1).sum()}"
    )
    print(
        f"Predicted Human:    "
        f"{(results_df['prediction'] == 0).sum()}"
    )

    print()
    print("AI PROBABILITY")
    print("-" * 70)

    print(f"Mean:               {ai_prob.mean() * 100:.2f}%")
    print(f"Median:             {ai_prob.median() * 100:.2f}%")
    print(f"Minimum:            {ai_prob.min() * 100:.2f}%")
    print(f"Maximum:            {ai_prob.max() * 100:.2f}%")

    print()
    print("THRESHOLDS")
    print("-" * 70)

    for threshold in [0.50, 0.70, 0.80, 0.90, 0.95]:

        count = (ai_prob >= threshold).sum()

        percentage = count / len(results_df) * 100

        print(
            f"AI >= {threshold * 100:5.0f}%: "
            f"{count:3d}/{len(results_df)} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("INDIVIDUAL PREDICTIONS")
    print("=" * 70)

    for _, row in results_df.iterrows():

        print()
        print(f"File:       {Path(row['file']).name}")
        print(f"Prediction: {row['prediction_name']}")
        print(
            f"AI prob:    "
            f"{row['ai_probability'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()