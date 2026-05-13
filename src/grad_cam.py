import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

from src.data_preprocessing import BrainTumorDataset, get_transforms, load_config
from src.models import ResNet18FineTuning

config = load_config()


# -------------------------------------------------
# Improved Grad-CAM
# -------------------------------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, class_idx=None):
        self.model.eval()

        output = self.model(input_image)

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        loss = output[:, class_idx]
        self.model.zero_grad()
        loss.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        # Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1).squeeze()

        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()

        # Resize to original image size (assuming 128x128)
        cam = cv2.resize(cam, (128, 128))

        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


# -------------------------------------------------
# Improved Visualization with Denormalization
# -------------------------------------------------
def overlay_cam(image_tensor, cam, save_path, mean=None, std=None):
    """
    image_tensor: torch tensor (C, H, W) in [0,1] or normalized
    """
    img = image_tensor.permute(1, 2, 0).cpu().numpy().copy()

    # Denormalize if mean/std provided
    if mean is not None and std is not None:
        img = img * std + mean

    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    # Overlay
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, overlay)
    print(f"Saved: {save_path}")


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18FineTuning(num_classes=config['num_classes']).to(device)
    model.load_state_dict(torch.load("models/resnet18_finetuning_best_balanced.pth", 
                                     map_location=device))
    model.eval()

    # === Try different layers if the current one is bad ===
    # Options: layer4[-1], layer4[-2], layer4[1], layer4[0]
    target_layer = model.model.layer4[-1]        # ← change this line if needed

    cam_generator = GradCAM(model.model, target_layer)

    # Get your test transforms
    transform = get_transforms(train=False)
    
    # Try to extract mean and std from transforms (for denormalization)
    mean = std = None
    for t in transform.transforms:
        if isinstance(t, transforms.Normalize):
            mean = t.mean
            std = t.std
            break

    dataset = BrainTumorDataset(
        root_dir=config['paths']['raw_data'],
        transform=transform,
        train=False
    )

    os.makedirs("results/gradcam", exist_ok=True)

    for i in range(10):   # Change range as needed
        image, label = dataset[i]
        input_tensor = image.unsqueeze(0).to(device)

        cam = cam_generator.generate(input_tensor)

        save_path = f"results/gradcam/sample_{i}_label_{label}.png"
        overlay_cam(image, cam, save_path, mean=mean, std=std)


if __name__ == "__main__":
    main()