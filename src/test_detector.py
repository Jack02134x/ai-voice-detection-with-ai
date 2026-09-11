import torch

from dataset import VoiceDataset
from model import AIVoiceDetector


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# Dataset
dataset = VoiceDataset("data/dataset.csv")

# Model
model = AIVoiceDetector()
model = model.to(device)
model.eval()


for i in range(len(dataset)):
    waveform, label = dataset[i]

    path = dataset.data.iloc[i]["file"]

    waveform = waveform.unsqueeze(0)
    waveform = waveform.to(device)

    with torch.no_grad():
        logits = model(waveform)

    probabilities = torch.softmax(logits, dim=1)

    human_probability = probabilities[0, 0].item()
    ai_probability = probabilities[0, 1].item()

    print()
    print(f"File: {path}")
    print(f"True label: {'AI' if label.item() == 1 else 'Human'}")
    print(f"Human probability: {human_probability:.4f}")
    print(f"AI probability:    {ai_probability:.4f}")