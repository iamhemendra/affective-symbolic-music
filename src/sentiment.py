import torch
from transformers import pipeline

device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    device=device
)

LABEL_TO_QUADRANT = {
    "joy": "Q1",
    "surprise": "Q1",
    "anger": "Q2",
    "fear": "Q2",
    "disgust": "Q2",
    "sadness": "Q3",
    "neutral": "Q4",
}

def map_text_to_quadrant(text: str) -> tuple[str, str, float]:
    """Analyzes text sentiment and routes to Russell's 4 Affect Quadrants."""
    results = classifier(text)[0]
    top_match = sorted(results, key=lambda x: x["score"], reverse=True)[0]
    dominant_label = top_match["label"].lower()
    score = top_match["score"]
    quadrant = LABEL_TO_QUADRANT.get(dominant_label, "Q4")
    return quadrant, dominant_label, score
