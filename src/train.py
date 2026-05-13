# Import PyTorch library
import torch

# Import neural network module
import torch.nn as nn

# Import optimization algorithms (Adam, SGD, etc.)
import torch.optim as optim

# DataLoader helps load data in batches during training
from torch.utils.data import DataLoader

# Allows modifying Python system paths
import sys


# Add current directory "." to Python path
# so Python can find the src/ folder imports
sys.path.append('.')


# Import custom dataset, transforms, and config loader
from src.data_preprocessing import (
    BrainTumorDataset,   # Custom dataset class
    get_transforms,      # Image preprocessing & augmentation
    load_config          # Loads settings from config file
)

# Import the ResNet18 fine-tuning model
from src.models import ResNet18FineTuning


# Load training configuration (batch size, epochs, paths, etc.)
config = load_config()


# -------------------------------------------------
# FUNCTION: Compute class weights
# -------------------------------------------------
# Used to handle class imbalance in the dataset
# Rare classes get higher weight during training
def compute_class_weights(labels, num_classes):

    # Count how many samples exist for each class
    counts = torch.bincount(torch.tensor(labels))

    # Inverse frequency weighting
    # Smaller classes => larger weight
    weights = 1.0 / (counts + 1e-6)

    # Normalize weights
    # Keeps overall scale balanced
    weights = weights / weights.sum() * num_classes

    return weights


# -------------------------------------------------
# MAIN TRAINING FUNCTION
# -------------------------------------------------
def main():

    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\n=== EXPERIMENT 3: RESNET18 FINE-TUNING (BALANCED) ===")
    print(f"Device: {device}")

    # -------------------------------------------------
    # DATASET CREATION
    # -------------------------------------------------

    # Training dataset
    train_dataset = BrainTumorDataset(
        root_dir=config['paths']['raw_data'],   # Dataset location
        transform=get_transforms(train=True),   # Data augmentation for training
        train=True
    )

    # Testing dataset
    test_dataset = BrainTumorDataset(
        root_dir=config['paths']['raw_data'],
        transform=get_transforms(train=False),  # No augmentation for testing
        train=False
    )

    # -------------------------------------------------
    # DATALOADERS
    # -------------------------------------------------

    # Loads training data in batches
    # shuffle=True randomizes data every epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    # Loads testing data in batches
    # shuffle=False keeps order fixed
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # -------------------------------------------------
    # MODEL SETUP
    # -------------------------------------------------

    # Create ResNet18 model for fine-tuning
    # Move model to GPU/CPU
    model = ResNet18FineTuning(
        num_classes=config['num_classes']
    ).to(device)

    # -------------------------------------------------
    # HANDLE CLASS IMBALANCE
    # -------------------------------------------------

    # Compute weights based on dataset imbalance
    class_weights = compute_class_weights(
        train_dataset.labels,
        config['num_classes']
    ).to(device)

    # CrossEntropyLoss is standard for classification
    # weight=class_weights penalizes mistakes on rare classes more
    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # -------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------

    # Adam optimizer updates model weights
    optimizer = optim.Adam(
        model.parameters(),   # Parameters to optimize
        lr=1e-4,              # Learning rate
        weight_decay=1e-4     # L2 regularization to reduce overfitting
    )

    # -------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------

    # Stores best accuracy achieved
    best_acc = 0

    # Loop over all epochs
    for epoch in range(config['epochs']):

        # Set model to training mode
        # Enables dropout/batchnorm training behavior
        model.train()

        # Variables for tracking metrics
        running_loss = 0
        correct = 0
        total = 0

        # Loop through batches
        for images, labels in train_loader:

            # Move images and labels to GPU/CPU
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients from previous batch
            optimizer.zero_grad()

            # Forward pass
            # Model predicts outputs
            outputs = model(images)

            # Compute loss between predictions and true labels
            loss = criterion(outputs, labels)

            # Backpropagation
            # Compute gradients
            loss.backward()

            # Update model weights
            optimizer.step()

            # Add batch loss to total loss
            running_loss += loss.item()

            # -------------------------------------------------
            # ACCURACY CALCULATION
            # -------------------------------------------------

            # Get predicted class index
            _, predicted = torch.max(outputs, 1)

            # Total number of samples
            total += labels.size(0)

            # Count correct predictions
            correct += (
                predicted == labels
            ).sum().item()

        # Average loss across batches
        avg_loss = running_loss / len(train_loader)

        # Compute accuracy percentage
        acc = 100 * correct / total

        # Print training progress
        print(
            f"Epoch [{epoch+1}/{config['epochs']}], "
            f"Loss: {avg_loss:.4f}, "
            f"Acc: {acc:.2f}%"
        )

        # -------------------------------------------------
        # SAVE BEST MODEL
        # -------------------------------------------------

        # Save model only if accuracy improves
        if acc > best_acc:

            best_acc = acc

            torch.save(
                model.state_dict(),   # Save learned weights only
                "models/resnet18_finetuning_best_balanced.pth"
            )

    # -------------------------------------------------
    # SAVE FINAL MODEL
    # -------------------------------------------------

    torch.save(
        model.state_dict(),
        "models/resnet18_finetuning_final_balanced.pth"
    )

    print("\n✅ Fine-tuning complete")
    print(f"🏆 Best Accuracy: {best_acc:.2f}%")


# -------------------------------------------------
# ENTRY POINT
# -------------------------------------------------

# Runs main() only if this file is executed directly
# Prevents automatic execution when imported elsewhere
if __name__ == "__main__":
    main()