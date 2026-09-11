import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import VoiceDataset
from model import AIVoiceDetector


TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-3

MODEL_OUTPUT = "models/ai_voice_detector/best_model.pt"


# -------------------------
# Device
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# -------------------------
# Datasets
# -------------------------

train_dataset = VoiceDataset(TRAIN_CSV)
val_dataset = VoiceDataset(VAL_CSV)

print(f"Training samples:   {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")


# -------------------------
# DataLoaders
# -------------------------

loader_kwargs = {
    "batch_size": BATCH_SIZE,
    "num_workers": 2,
    "pin_memory": device.type == "cuda",
}

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=device.type == "cuda",
)

# -------------------------
# Model
# -------------------------

model = AIVoiceDetector(freeze_encoder=True)
model = model.to(device)


# -------------------------
# Loss + optimizer
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
)


# -------------------------
# Training
# -------------------------

best_val_accuracy = 0.0


for epoch in range(EPOCHS):

    # =====================
    # Training
    # =====================

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for waveforms, labels in train_loader:

        waveforms = waveforms.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(waveforms)

        loss = criterion(logits, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    train_loss = total_loss / len(train_loader)


    # =====================
    # Validation
    # =====================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.inference_mode():

        for waveforms, labels in val_loader:

            waveforms = waveforms.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            logits = model(waveforms)

            loss = criterion(logits, labels)

            val_loss += loss.item()

            predictions = logits.argmax(dim=1)

            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total
    val_loss = val_loss / len(val_loader)


    # =====================
    # Print results
    # =====================

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"Loss: {train_loss:.4f} "
        f"Train: {train_accuracy * 100:.2f}% "
        f"Val Loss: {val_loss:.4f} "
        f"Val: {val_accuracy * 100:.2f}%"
    )


    # =====================
    # Save best model
    # =====================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy,
            },
            MODEL_OUTPUT,
        )

        print(
            f"  -> Saved best model "
            f"(validation accuracy: "
            f"{val_accuracy * 100:.2f}%)"
        )


print()
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)
print(f"Model saved to: {MODEL_OUTPUT}")