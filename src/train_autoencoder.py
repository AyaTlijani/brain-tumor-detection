import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.autoencoder import Autoencoder
from src.data_preprocessing import BrainTumorDataset, get_transforms, load_config

# =========================
# CONFIG
# =========================
config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = config['paths']['raw_data']
SAVE_PATH = Path(r"C:\Users\MSI\Desktop\Aya-Tlijani-Brain-Tumor\models\autoencoder.pth")

# =========================
# DATASET
# =========================
dataset = BrainTumorDataset(
    root_dir=DATA_PATH,
    transform=get_transforms(train=False),
    train=False
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# MODEL
# =========================
model = Autoencoder().to(device)

# =========================
# 🔥 LOSS (IMPORTANT UPGRADE)
# =========================
l1 = nn.L1Loss()

def feature_loss(x, y):
    # encourages structural similarity (better than pixel-only)
    return torch.mean((x - y) ** 2)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# =========================
# TRAINING
# =========================
epochs = 20

print("\n🚀 Training STRONG Autoencoder...\n")

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        outputs = model(images)

        loss = (
            0.7 * l1(outputs, images) +
            0.3 * feature_loss(outputs, images)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(loader):.6f}")

# =========================
# SAVE
# =========================
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)

print("\n✔ Saved:", SAVE_PATH)