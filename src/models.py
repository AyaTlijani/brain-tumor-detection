# Import PyTorch
import torch

# Import neural network module
import torch.nn as nn

# Import pretrained computer vision models
from torchvision import models

# Import pretrained weights for ResNet18
from torchvision.models import ResNet18_Weights


# ======================================================
# EXPERIMENT 1: BASELINE CNN
# ======================================================
# A simple custom CNN built from scratch
class BaselineCNN(nn.Module):

    # num_classes = number of output categories
    # Default is 4 classes
    def __init__(self, num_classes=4):

        # Initialize parent nn.Module class
        super().__init__()

        # -------------------------------------------------
        # FEATURE EXTRACTION PART
        # -------------------------------------------------
        # Learns visual patterns from images
        self.features = nn.Sequential(

            # First convolution layer
            # Input: 3 channels (RGB image)
            # Output: 32 feature maps
            nn.Conv2d(3, 32, kernel_size=3, padding=1),

            # Activation function
            nn.ReLU(),

            # Reduce image size by half
            nn.MaxPool2d(2),

            # Second convolution layer
            # Input: 32 channels
            # Output: 64 feature maps
            nn.Conv2d(32, 64, kernel_size=3, padding=1),

            # Activation
            nn.ReLU(),

            # Downsampling again
            nn.MaxPool2d(2)
        )

        # -------------------------------------------------
        # CLASSIFICATION PART
        # -------------------------------------------------
        # Converts extracted features into class predictions
        self.classifier = nn.Sequential(

            # Convert 3D feature maps into 1D vector
            nn.Flatten(),

            # Fully connected layer
            # 64 * 32 * 32 depends on image size after pooling
            nn.Linear(64 * 32 * 32, 128),

            # Activation
            nn.ReLU(),

            # Randomly disables neurons during training
            # Helps reduce overfitting
            nn.Dropout(0.3),

            # Final output layer
            # Produces class scores
            nn.Linear(128, num_classes)
        )

    # -------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------
    # Defines how data moves through the network
    def forward(self, x):

        # Extract features
        x = self.features(x)

        # Classify features
        return self.classifier(x)


# ======================================================
# EXPERIMENT 2: FEATURE EXTRACTION
# ======================================================
# Uses pretrained ResNet18 as a fixed feature extractor
class ResNet18FeatureExtractor(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # -------------------------------------------------
        # LOAD PRETRAINED RESNET18
        # -------------------------------------------------
        # Uses ImageNet pretrained weights
        self.model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # -------------------------------------------------
        # FREEZE ALL LAYERS
        # -------------------------------------------------
        # Prevent pretrained weights from updating
        # Model acts only as a feature extractor
        for param in self.model.parameters():
            param.requires_grad = False

        # Number of inputs going into final FC layer
        in_features = self.model.fc.in_features

        # -------------------------------------------------
        # REPLACE FINAL CLASSIFIER
        # -------------------------------------------------
        # Original ResNet classifier is replaced
        # with one adapted for brain tumor classes
        self.model.fc = nn.Sequential(

            # First fully connected layer
            nn.Linear(in_features, 256),

            # Activation
            nn.ReLU(),

            # Regularization
            nn.Dropout(0.4),

            # Final classification layer
            nn.Linear(256, num_classes)
        )

    # Forward propagation
    def forward(self, x):
        return self.model(x)


# ======================================================
# EXPERIMENT 3: FINE-TUNING
# ======================================================
# Similar to feature extraction
# BUT some deep layers are trainable
class ResNet18FineTuning(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        # -------------------------------------------------
        # LOAD PRETRAINED RESNET18
        # -------------------------------------------------
        self.model = models.resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # -------------------------------------------------
        # FREEZE ALL LAYERS FIRST
        # -------------------------------------------------
        for param in self.model.parameters():
            param.requires_grad = False

        # -------------------------------------------------
        # UNFREEZE DEEPER LAYERS
        # -------------------------------------------------
        # layer3 and layer4 learn more task-specific features
        # These layers will continue learning
        for param in self.model.layer3.parameters():
            param.requires_grad = True

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Get input size for final FC layer
        in_features = self.model.fc.in_features

        # -------------------------------------------------
        # REPLACE FINAL CLASSIFIER
        # -------------------------------------------------
        self.model.fc = nn.Sequential(

            # Fully connected layer
            nn.Linear(in_features, 256),

            # Activation
            nn.ReLU(),

            # Reduce overfitting
            nn.Dropout(0.4),

            # Final class predictions
            nn.Linear(256, num_classes)
        )

    # -------------------------------------------------
    # FORWARD PASS
    # -------------------------------------------------
    def forward(self, x):

        # Send input through ResNet18
        return self.model(x)