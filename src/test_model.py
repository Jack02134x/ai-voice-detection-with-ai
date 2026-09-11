import torch
from model import AIVoiceDetector


MODEL_PATH = "models/ai_voice_detector/best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLIP_SAMPLES = 16_000 * 4


def main():
    print("=" * 60)
    print("AI Voice Detector - Model Sanity Test")
    print("=" * 60)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA: {torch.version.cuda}")
    else:
        print("WARNING: CUDA is not available.")

    print()
    print("Loading local Wav2Vec2 + detector...")

    model = AIVoiceDetector(freeze_encoder=True)

    print("✓ Local Wav2Vec2 loaded")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print("✓ Detector checkpoint loaded")
    print(f"✓ Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"✓ Checkpoint validation accuracy: "
          f"{checkpoint.get('val_accuracy', 'unknown')}")

    model.to(DEVICE)
    model.eval()

    print()
    print("Creating dummy 4-second waveform...")

    # Fake 4-second mono 16 kHz audio
    waveform = torch.randn(1, CLIP_SAMPLES).to(DEVICE)

    print(f"Input shape: {tuple(waveform.shape)}")

    print()
    print("Running forward pass...")

    with torch.no_grad():
        logits = model(waveform)
        probabilities = torch.softmax(logits, dim=1)

    print("✓ Forward pass successful")

    print()
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Probabilities shape: {tuple(probabilities.shape)}")

    human_probability = probabilities[0, 0].item()
    ai_probability = probabilities[0, 1].item()

    prediction = torch.argmax(probabilities, dim=1).item()

    print()
    print("Prediction:")
    print(f"  Human: {human_probability:.6f}")
    print(f"  AI:    {ai_probability:.6f}")
    print(f"  Class: {'AI' if prediction == 1 else 'Human'}")

    print()
    print("=" * 60)
    print("✓ MODEL SANITY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()