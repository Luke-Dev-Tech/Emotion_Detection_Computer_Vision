from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# ── Emotion labels (FER-2013 order) ──────────────────────────────────────────
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ── Emotion accent colours (returned in JSON for canvas drawing) ──────────────
EMOTION_COLORS = {
    'Angry':    '#ff4757',
    'Disgust':  '#2ed573',
    'Fear':     '#a29bfe',
    'Happy':    '#ffd32a',
    'Neutral':  '#74b9ff',
    'Sad':      '#636e72',
    'Surprise': '#fd79a8',
}

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load emotion model ────────────────────────────────────────────────────────
# Tries full-model load first (torch.save(model, path)).
# If that fails (state-dict only), falls back to a standard
# ResNet-18 head that matches the FER-2013 augmented checkpoint.
MODEL_PATH = "FaceEmotionDection_argumented_final.pth"

def _build_efficientnet_fer():
    """Minimal EfficientNet-B2 head for 7-class FER-2013."""
    import torchvision.models as tv
    
    # 1. Load the empty EfficientNet-B2 architecture
    m = tv.efficientnet_b2(weights=None)
    
    # # 2. Modify the first layer to accept 1-channel grayscale instead of 3-channel RGB
    # # EfficientNet's first conv layer is stored in features[0][0]
    # original_conv = m.features[0][0]
    # m.features[0][0] = nn.Conv2d(
    #     in_channels=1, 
    #     out_channels=original_conv.out_channels, 
    #     kernel_size=original_conv.kernel_size, 
    #     stride=original_conv.stride, 
    #     padding=original_conv.padding, 
    #     bias=False
    # )
    
    # 3. Modify the final classification layer for our 7 emotions
    # EfficientNet-B2 outputs 1408 features before the final layer
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, len(EMOTIONS))
    
    return m

_loaded = torch.load(MODEL_PATH, map_location=device, weights_only=False)

if isinstance(_loaded, dict):
    # It's a state-dict (OrderedDict) — build the architecture and load weights
    print(f"[INFO] Detected state-dict in {MODEL_PATH}, building EfficientNet B2 backbone …")
    emotion_model = _build_efficientnet_fer()    
    state = {k.replace("module.", ""): v for k, v in _loaded.items()}
    emotion_model.load_state_dict(state, strict=False)
    print("[INFO] State-dict loaded into EfficientNet B2 backbone")
else:
    # Full model object (nn.Module subclass)
    emotion_model = _loaded
    print(f"[INFO] Loaded full model object from {MODEL_PATH}")

emotion_model.to(device)
emotion_model.eval()

# ── YOLO (person detection only) ──────────────────────────────────────────────
yolo = YOLO("yolov8n.pt")
PERSON_CLASS = 0   # COCO class 0 = person

# ── OpenCV face detector ──────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Input transform (FER-2013: 48×48 grayscale, normalised) ──────────────────
emotion_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])




# ── Inference helper ──────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
def predict_emotion(face_pil: Image.Image):
    """Return (emotion_label, confidence_float)."""
    with torch.inference_mode():
        t = emotion_transform(face_pil).unsqueeze(0).to(device)
        logits = emotion_model(t)
        probs  = torch.softmax(logits, dim=1)
        idx    = torch.argmax(probs, dim=1).item()
        conf   = probs.max().item() * 100
        # # --- MATPLOTLIB FIX ---
        # # 1. Remove batch dimension (squeeze)
        # # 2. Move to CPU (Matplotlib can't read GPU tensors)
        # # 3. Rearrange axes from (C, H, W) to (H, W, C)
        # img_for_plt = t.squeeze().cpu().permute(1, 2, 0).numpy()
        
        # # 4. De-normalize (undo the mean=0.5, std=0.5) so it isn't completely dark
        # img_for_plt = (img_for_plt * 0.5) + 0.5
        
        # # 5. Show image (removed cmap='gray' because it is now 3 channels)
        # plt.imshow(img_for_plt)
        # plt.title(f"Model Input - {EMOTIONS[idx]}")
        # plt.show()
        
        # plt.imshow(t.squeeze().cpu().permute(1, 2, 0).numpy(), cmap='gray')
        # plt.title("Model Input")
        # plt.show()
        # ----------------------
    return EMOTIONS[idx], round(conf, 2)


def detect_faces_and_emotions(img_pil: Image.Image):
    """
    Pipeline:
      1. YOLO → person bounding boxes only
      2. Inside each person crop → OpenCV Haar face detection
      3. Emotion inference on each face crop
    Returns list of dicts with box coords + emotion data.
    """
    img_np = np.array(img_pil.convert("RGB"))
    results_yolo = yolo(img_np, classes=[PERSON_CLASS], verbose=False)[0]

    boxes_out = []

    for box in results_yolo.boxes:
        cls_id = int(box.cls[0])
        if cls_id != PERSON_CLASS:
            continue

        px1, py1, px2, py2 = map(int, box.xyxy[0])
        # Clamp to image bounds
        H, W = img_np.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(W, px2), min(H, py2)

        person_crop = img_np[py1:py2, px1:px2]
        if person_crop.size == 0:
            continue

        gray = cv2.cvtColor(person_crop, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )

        # if len(faces) == 0:
        #     # Fallback: treat upper-third of person box as face region
        #     fh = (py2 - py1) // 3
        #     faces = [(0, 0, px2 - px1, fh)]
        
        if len(faces) == 0:
            continue

        for (fx, fy, fw, fh) in faces:
            # Absolute coords in the full image
            ax1 = px1 + fx
            ay1 = py1 + fy
            ax2 = ax1 + fw
            ay2 = ay1 + fh

            face_pil = Image.fromarray(img_np[ay1:ay2, ax1:ax2])
            if face_pil.width < 8 or face_pil.height < 8:
                continue

            emotion, confidence = predict_emotion(face_pil)

            if confidence < 70:
                # emotion = "Uncertain"
                pass
            
            boxes_out.append({
                "x1": int(ax1),
                "y1": int(ay1),
                "x2": int(ax2),
                "y2": int(ay2),
                "emotion": emotion,
                "confidence": float(confidence),
                "color": EMOTION_COLORS.get(emotion, "#ffffff"),
            })

    return boxes_out


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/guide")
def guide():
    return render_template("guide.html")

@app.route("/live-cam")
def livecam():
    return render_template("livecam.html")

# ── Static image upload ───────────────────────────────────────────────────────
@app.route("/", methods=["POST"])
def predict_upload():
    file = request.files.get("image")
    if not file or file.filename == "":
        return render_template("index.html", error="No image selected.")

    img_pil = Image.open(file).convert("RGB")
    detections = detect_faces_and_emotions(img_pil)

    # Encode image for preview
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return render_template(
        "index.html",
        detections=detections,
        img_data=img_b64,
    )


# ── Real-time webcam endpoint ─────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_realtime():
    file = request.files.get("image")
    if not file:
        return jsonify({"error": "No image"}), 400

    img_pil = Image.open(file.stream).convert("RGB")
    detections = detect_faces_and_emotions(img_pil)

    return jsonify({"boxes": detections})


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5300, debug=True)