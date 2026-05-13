import torch
import numpy as np
import pandas as pd

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score
)

from src.data_preprocessing import BrainTumorDataset, get_transforms, load_config
from src.models import ResNet18FeatureExtractor, ResNet18FineTuning

config = load_config()


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
def load_model(model_class, path, device):

    model = model_class(num_classes=config['num_classes']).to(device)

    state_dict = torch.load(path, map_location=device)

    model.load_state_dict(state_dict)

    model.eval()

    return model


# -------------------------------------------------
# EVALUATION
# -------------------------------------------------
def evaluate_model(model, loader, device):

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")

    return acc, precision, recall, f1, all_labels, all_preds


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== MODEL COMPARISON (IMPROVED) ===")

    test_dataset = BrainTumorDataset(
        root_dir=config['paths']['raw_data'],
        transform=get_transforms(train=False),
        train=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    models_to_test = [
        ("Feature Extraction", ResNet18FeatureExtractor,
         "models/resnet18_feature_extraction_best.pth"),

        ("Fine Tuning", ResNet18FineTuning,
         "models/resnet18_finetuned_best.pth"),

        ("Fine Tuning Balanced", ResNet18FineTuning,
         "models/resnet18_finetuning_best_balanced.pth"),
    ]

    results = []

    for name, model_class, path in models_to_test:

        print(f"\nEvaluating: {name}")

        model = load_model(model_class, path, device)

        acc, precision, recall, f1, y_true, y_pred = evaluate_model(
            model, test_loader, device
        )

        print(f"Accuracy   : {acc:.4f}")
        print(f"Precision  : {precision:.4f}")
        print(f"Recall     : {recall:.4f}")
        print(f"F1-score   : {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_true,
            y_pred,
            target_names=config['class_names'],
            digits=4
        ))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1
        })

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------
    df = pd.DataFrame(results)

    df.to_csv("results/model_comparison.csv", index=False)

    print("\n✅ Saved comparison to: results/model_comparison.csv")
    print(df)


if __name__ == "__main__":
    main()