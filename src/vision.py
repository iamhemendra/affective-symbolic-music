import torch
import numpy as np
from PIL import Image
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
fer_classifier = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=device
)

FER_TO_QUADRANT = {
    "happy": "Q1",
    "surprise": "Q1",
    "angry": "Q2",
    "fear": "Q2",
    "disgust": "Q2",
    "sad": "Q3",
    "neutral": "Q4",
}

def analyze_facial_affect(frame_numpy: np.ndarray) -> tuple[str, str, float]:
    """Processes webcam image and classifies into one of 7 canonical emotions."""
    if frame_numpy is None:
        return "neutral", "Q4", 0.0

    if frame_numpy.dtype != np.uint8:
        frame_numpy = frame_numpy.astype(np.uint8)

    pil_img = Image.fromarray(frame_numpy)
    predictions = fer_classifier(pil_img)
    top_pred = predictions[0]
    
    emotion = top_pred["label"].lower()
    score = top_pred["score"]
    
    if score < 0.35:
        return "neutral", "Q4", score

    quadrant = FER_TO_QUADRANT.get(emotion, "Q4")
    return emotion, quadrant, score
