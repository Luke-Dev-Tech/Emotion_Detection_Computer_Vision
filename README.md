# Face Emotion Detection

A real-time facial emotion recognition web application trained on the **FER-2013** dataset using a fine-tuned **EfficientNet-B2** backbone. Detects faces in images or live webcam streams and classifies them into 7 emotions.

---

## Demo

| Mode | Description |
|---|---|
| Image Upload | Upload any photo — faces are detected and labeled instantly |
| Live Webcam | Real-time emotion detection from your browser camera |

---

## Emotions Detected

`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

---

## How It Works

Three-stage inference pipeline:

```
Input Image
    │
    ▼
YOLOv8n  ──────────────────── detects person bounding boxes
    │
    ▼
OpenCV Haar Cascade ────────── detects faces within each person crop
    │
    ▼
EfficientNet-B2 ─────────────  classifies emotion (7 classes)
    │
    ▼
Labeled result with confidence score & color-coded box
```

---

## Model Training

Trained in **5 progressive fine-tuning phases** on FER-2013 (35,887 images):

| Phase | Unfrozen Layers | LR | Epochs |
|---|---|---|---|
| 1 | Classifier head only | 0.001 | 15 |
| 2 | Last 1 backbone block | 0.0001 | 10 |
| 3 | Last 3 backbone blocks | 0.00001 | 15 |
| 4 | Last 5 backbone blocks | 0.000005 | 15 |
| 5 | Full model | 0.000001 | 20 |

**Techniques used:**
- Class imbalance handled with inverse-frequency class weights
- Data augmentation: random flip, rotation, affine shift, color jitter, random erasing
- `CosineAnnealingLR` scheduler per phase
- Early stopping with best-weight restoration

---

## Tech Stack

- **Model**: PyTorch · EfficientNet-B2 (torchvision)
- **Detection**: YOLOv8n (Ultralytics) · OpenCV Haar Cascade
- **Web**: Flask
- **Training**: Google Colab · FER-2013 (Kaggle)

---

## Getting Started

**1. Clone the repo**
```bash
git clone https://github.com/Luke-Dev-Tech/Emotion_Detection_Computer_Vision.git
cd Emotion_Detection_Computer_Vision
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the model weights**

Place `FaceEmotionDection_argumented_final.pth` inside the `EmotionDection/` folder.

**4. Run the app**
```bash
cd EmotionDection
python app.py
```

Open `http://localhost:5300` in your browser.

---

## Project Structure

```
├── FACE_EMOTION.ipynb          # Full training notebook (Colab)
├── requirements.txt
├── EmotionDection/
│   ├── app.py                  # Flask app
│   └── templates/
│       ├── index.html          # Image upload page
│       └── livecam.html        # Real-time webcam page
```

---

## Dataset

[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) — 35,887 grayscale 48×48 face images across 7 emotion classes.

---

## License

[MIT](LICENSE)
