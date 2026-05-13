import torch

from torch.utils.data import Dataset
from torchvision import transforms

from pathlib import Path

import yaml
from PIL import Image
import numpy as np


# -----------------------------------
# LOAD CONFIG
# -----------------------------------
def load_config():

    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


config = load_config()


# -----------------------------------
# DATASET
# -----------------------------------
class BrainTumorDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
        train=True
    ):

        self.root_dir = Path(root_dir)

        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.class_names = config['class_names']

        split = "Training" if train else "Testing"

        base_path = self.root_dir / split

        # -----------------------------------
        # LOAD IMAGES
        # -----------------------------------
        for label_idx, class_name in enumerate(self.class_names):

            class_dir = base_path / class_name

            if class_dir.exists():

                for ext in ["*.jpg", "*.png", "*.jpeg"]:

                    for img_path in class_dir.glob(ext):

                        self.image_paths.append(img_path)
                        self.labels.append(label_idx)

        print(f"Loaded {len(self.image_paths)} images from {split} set")

        print(
            f"Class distribution: "
            f"{np.bincount(self.labels)}"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]

        image = Image.open(img_path).convert("RGB")

        label = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        # -----------------------------------
        # APPLY TRANSFORMS
        # -----------------------------------
        if self.transform:
            image = self.transform(image)

        return image, label


# -----------------------------------
# TRANSFORMS
# -----------------------------------
def get_transforms(train=True):

    # -----------------------------------
    # REQUIRED FOR PRETRAINED RESNET
    # -----------------------------------
    imagenet_mean = [0.485, 0.456, 0.406]

    imagenet_std = [0.229, 0.224, 0.225]

    # -----------------------------------
    # TRAIN TRANSFORMS
    # -----------------------------------
    if train and config.get(
        'augmentation',
        {}
    ).get('enabled', True):

        return transforms.Compose([

            transforms.Resize(
                (
                    config['image_size'],
                    config['image_size']
                )
            ),

            # -----------------------------------
            # DATA AUGMENTATION
            # -----------------------------------
            transforms.RandomRotation(15),

            transforms.RandomHorizontalFlip(),

            transforms.RandomVerticalFlip(),

            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05)
            ),

            transforms.ToTensor(),

            # -----------------------------------
            # NORMALIZATION
            # -----------------------------------
            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std
            )
        ])

    # -----------------------------------
    # TEST TRANSFORMS
    # -----------------------------------
    else:

        return transforms.Compose([

            transforms.Resize(
                (
                    config['image_size'],
                    config['image_size']
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=imagenet_mean,
                std=imagenet_std
            )
        ])


# -----------------------------------
# TEST
# -----------------------------------
if __name__ == "__main__":

    train_dataset = BrainTumorDataset(
        root_dir=config['paths']['raw_data'],
        transform=get_transforms(train=True),
        train=True
    )

    print("✅ Dataset ready!")