import torch
import numpy as np

from src.autoencoder import Autoencoder
from src.data_preprocessing import BrainTumorDataset, get_transforms, load_config

config = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
model = Autoencoder().to(device)
model.load_state_dict(torch.load("models/autoencoder.pth", map_location=device))
model.eval()

# =========================
# DATASET
# =========================
dataset = BrainTumorDataset(
    root_dir=config['paths']['raw_data'],
    transform=get_transforms(train=False),
    train=False
)

loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

# =========================
# ERROR FUNCTION
# =========================
def error(x, recon):
    return torch.mean((x - recon) ** 2).item()

errors = []
labels = []

print("\n=== EVALUATION ===\n")

with torch.no_grad():
    for i, (img, label) in enumerate(loader):

        img = img.to(device)
        recon = model(img)

        e = error(img, recon)

        errors.append(e)
        labels.append(label.item())

# =========================
# NORMAL STATISTICS
# =========================
errors = np.array(errors)
labels = np.array(labels)

normal_errors = errors[labels == 0]
tumor_errors = errors[labels != 0]

mean_n = normal_errors.mean()
std_n = normal_errors.std()

threshold = mean_n + 2 * std_n

# =========================
# METRICS
# =========================
preds = errors > threshold
accuracy = np.mean(preds == (labels != 0))

print("NORMAL_MEAN", mean_n)
print("NORMAL_STD", std_n)
print("TUMOR_MEAN", tumor_errors.mean())
print("TUMOR_STD", tumor_errors.std())
print("THRESHOLD", threshold)
print("ACCURACY", accuracy)

separation = abs(tumor_errors.mean() - normal_errors.mean()) / (std_n + 1e-8)
print("SEPARATION_SCORE", separation)