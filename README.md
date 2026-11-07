# Brain Tumor MRI Classification

**Classifies brain MRI scans into four classes (no tumor, glioma, meningioma, pituitary tumor) and shows where the model looked, with Grad-CAM.**

A ResNet-18 is fine-tuned on brain MRI images, compared with a simpler feature-extraction approach, and served in a Streamlit dashboard. A second dashboard page classifies breast histopathology images.

> Educational project, not a medical device.

## Results

Test set of 1,600 MRI images, 4 classes:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| ResNet-18, feature extraction (frozen backbone) | 80.4% | 80.1% | 80.4% | 79.7% |
| ResNet-18, fine-tuned | 88.8% | 90.2% | 88.8% | 88.2% |
| ResNet-18, fine-tuned with class-weighted loss | **94.2%** | **94.7%** | **94.2%** | **94.1%** |

Training the whole network instead of only the last layer adds 8 points. Weighting the loss so mistakes on rarer classes cost more adds another 5.

Full table: [`results/model_comparison.csv`](results/model_comparison.csv). Grad-CAM examples: [`results/gradcam/`](results/gradcam).

## How it works

- **Data** ([`src/data_preprocessing.py`](src/data_preprocessing.py)): MRI images resized to 128×128, with augmentation on the training set.
- **Models** ([`src/models.py`](src/models.py)): ResNet-18 pretrained on ImageNet, either frozen with a new classifier head (feature extraction) or fully fine-tuned. A small CNN baseline and a convolutional autoencoder are also included.
- **Training** ([`src/train.py`](src/train.py)): class weights computed from the training labels. The checkpoint with the best test accuracy is kept; there is no separate validation split, so the reported scores are slightly optimistic.
- **Explainability** ([`src/grad_cam.py`](src/grad_cam.py)): Grad-CAM heatmaps over the last convolutional layer.
- **Dashboard** ([`dashboard/app.py`](dashboard/app.py)): upload a scan to get the predicted class, the confidence, the top-2 classes and the Grad-CAM overlay.

## Run it

```bash
pip install -r requirements.txt streamlit opencv-python
streamlit run dashboard/app.py
```

Trained weights are in [`models/`](models). To retrain, run `python -m src.train` from the repository root; settings are in [`config.yaml`](config.yaml).

## Stack

Python · PyTorch · torchvision · scikit-learn · Grad-CAM · Streamlit · OpenCV
