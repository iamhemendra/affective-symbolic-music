"""
Evaluation suite for Emotion-Conditioned Symbolic Music Generation.

Measures accuracy at three levels and writes results/evaluation.json,
which the web app (src/accuracy_ui.py) reads and displays.

    Level 1  Perception      : does the system read the user's emotion correctly?
                               (face: 7-emotion + quadrant, text: quadrant)
    Level 2  Conditioning    : does the generated music actually sound like the
                               requested quadrant? (independent classifier)
    Level 3  End-to-end      : input -> quadrant -> music -> judged quadrant
                               matches the ground-truth quadrant.

Install extras:   pip install scikit-learn pretty_midi pillow

Example:
    python -m src.evaluate \
        --face_dir data/eval/faces \
        --text_csv data/eval/text.csv \
        --emopia_dir data/EMOPIA/midi \
        --n_per_quadrant 25 --max_e2e 40
"""
import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
# Russell's circumplex: sign of (valence, arousal) for each quadrant
QUADRANT_VA = {"Q1": (1, 1), "Q2": (-1, 1), "Q3": (-1, -1), "Q4": (1, -1)}

# Ground-truth 7-emotion -> quadrant mapping.
# IMPORTANT: keep this identical to the mapping used inside src/vision.py.
EMOTION_TO_QUADRANT = {
    "happy": "Q1", "surprise": "Q1",
    "angry": "Q2", "fear": "Q2", "disgust": "Q2",
    "sad": "Q3",
    "neutral": "Q4",
}

# Krumhansl-Kessler key profiles (for major/minor "mode" feature)
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def find_midi_files(root):
    """All .mid/.midi files under root, de-duplicated (Windows globbing is
    case-insensitive, so '*.mid' and '*.MID' would otherwise match twice)."""
    found = set()
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.lower().endswith((".mid", ".midi")):
                found.add(os.path.join(dirpath, n))
    return sorted(found)


def norm_q(q):
    """Normalise labels such as 'Q1', 'q1 (happy)' -> 'Q1'."""
    s = str(q).strip().upper()
    for k in QUADRANTS:
        if s.startswith(k):
            return k
    return s


# --------------------------------------------------------------------------
# MIDI feature extraction (used by the independent quadrant classifier)
# --------------------------------------------------------------------------
def midi_features(path):
    """Return a fixed-length feature vector, or None if the MIDI is invalid/empty."""
    import pretty_midi

    try:
        pm = pretty_midi.PrettyMIDI(str(path))
    except Exception:
        return None
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    if len(notes) < 8:
        return None

    dur = max(pm.get_end_time(), 1e-6)
    pitches = np.array([n.pitch for n in notes], dtype=float)
    vel = np.array([n.velocity for n in notes], dtype=float)
    starts = np.sort(np.array([n.start for n in notes]))
    ioi = np.diff(starts)
    ioi = ioi[ioi > 1e-4]

    hist = np.zeros(12)
    for n in notes:
        hist[n.pitch % 12] += max(n.end - n.start, 1e-3)
    hist /= hist.sum()
    maj = max(np.nan_to_num(np.corrcoef(np.roll(_MAJOR, k), hist)[0, 1]) for k in range(12))
    mnr = max(np.nan_to_num(np.corrcoef(np.roll(_MINOR, k), hist)[0, 1]) for k in range(12))

    _, tempi = pm.get_tempo_changes()
    tempo = float(np.mean(tempi)) if len(tempi) else 120.0
    polyphony = sum(n.end - n.start for n in notes) / dur

    return [
        tempo,                       # tempo (BPM)
        len(notes) / dur,            # note density (arousal cue)
        vel.mean(), vel.std(),       # loudness / dynamics (arousal cue)
        pitches.mean(), pitches.std(), pitches.max() - pitches.min(),
        maj - mnr,                   # major-vs-minor tendency (valence cue)
        polyphony,
        float(ioi.mean()) if len(ioi) else 0.0,
        float(np.median(ioi)) if len(ioi) else 0.0,
    ]


def train_music_classifier(emopia_dir, seed=0):
    """Train an independent MIDI -> quadrant classifier on real EMOPIA clips.

    EMOPIA files are named like 'Q1_xxxxxxxx_0.mid', so the first two
    characters of the filename are the ground-truth quadrant.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    files = find_midi_files(emopia_dir)
    X, y = [], []
    for f in files:
        q = norm_q(Path(f).name[:2])
        if q not in QUADRANTS:
            continue
        feats = midi_features(f)
        if feats is not None:
            X.append(feats)
            y.append(q)
    if len(X) < 40:
        raise RuntimeError(f"Only {len(X)} usable EMOPIA clips found in {emopia_dir}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed)
    clf.fit(Xtr, ytr)
    ceiling = float(np.mean(clf.predict(Xte) == np.array(yte)))
    return clf, ceiling, len(X)


# --------------------------------------------------------------------------
# Level 1: perception
# --------------------------------------------------------------------------
def eval_faces(face_dir, analyze_facial_affect, max_per_class=50, seed=0):
    """face_dir/<emotion>/*.jpg|png  (FER-2013 folder layout works as-is)."""
    from PIL import Image
    from sklearn.metrics import f1_score

    rng = random.Random(seed)
    emo_true, emo_pred, q_true, q_pred = [], [], [], []
    for emo in EMOTION_TO_QUADRANT:
        folder = Path(face_dir) / emo
        imgs = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng.shuffle(imgs)
        for p in imgs[:max_per_class]:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                e, q, _ = analyze_facial_affect(arr)
            except Exception:
                e, q = "error", "error"
            emo_true.append(emo)
            emo_pred.append(str(e).strip().lower())
            q_true.append(EMOTION_TO_QUADRANT[emo])
            q_pred.append(norm_q(q))
    if not emo_true:
        return None, []
    res = {
        "n": len(emo_true),
        "emotion_accuracy": float(np.mean(np.array(emo_true) == np.array(emo_pred))),
        "emotion_macro_f1": float(f1_score(emo_true, emo_pred, average="macro", zero_division=0)),
        "quadrant_accuracy": float(np.mean(np.array(q_true) == np.array(q_pred))),
    }
    return res, list(zip(q_true, q_pred))


def eval_text(text_csv, map_text_to_quadrant, max_n=300, seed=0):
    """CSV with columns: text,quadrant   (quadrant in Q1..Q4)."""
    import csv
    from sklearn.metrics import f1_score

    rows = list(csv.DictReader(open(text_csv, encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    rows = rows[:max_n]
    q_true, q_pred = [], []
    for r in rows:
        try:
            q, _, _ = map_text_to_quadrant(r["text"])
            q = norm_q(q)
        except Exception:
            q = "error"
        q_true.append(norm_q(r["quadrant"]))
        q_pred.append(q)
    if not q_true:
        return None, []
    res = {
        "n": len(q_true),
        "quadrant_accuracy": float(np.mean(np.array(q_true) == np.array(q_pred))),
        "quadrant_macro_f1": float(f1_score(q_true, q_pred, average="macro", zero_division=0)),
    }
    return res, list(zip(q_true, q_pred))


# --------------------------------------------------------------------------
# Level 2: conditioning fidelity of the generator
# --------------------------------------------------------------------------
def _score_generation(y_true, y_pred, valid, total):
    """y_pred entries are None for invalid/failed generations (counted wrong)."""
    from sklearn.metrics import confusion_matrix

    correct = [t == p for t, p in zip(y_true, y_pred)]
    val_ok = [p is not None and QUADRANT_VA[t][0] == QUADRANT_VA[p][0] for t, p in zip(y_true, y_pred)]
    aro_ok = [p is not None and QUADRANT_VA[t][1] == QUADRANT_VA[p][1] for t, p in zip(y_true, y_pred)]
    per_q = {}
    for q in QUADRANTS:
        flags = [c for t, c in zip(y_true, correct) if t == q]
        per_q[q] = float(np.mean(flags)) if flags else None
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    cm = confusion_matrix([t for t, _ in pairs], [p for _, p in pairs], labels=QUADRANTS).tolist() if pairs else None
    return {
        "n": total,
        "valid_midi_rate": valid / total if total else None,
        "conditioning_accuracy": float(np.mean(correct)) if correct else None,
        "valence_accuracy": float(np.mean(val_ok)) if val_ok else None,
        "arousal_accuracy": float(np.mean(aro_ok)) if aro_ok else None,
        "per_quadrant": per_q,
        "confusion": cm,
    }


def eval_generation(clf, generate_track, n_per_quadrant=25):
    y_true, y_pred = [], []
    valid = 0
    for q in QUADRANTS:
        for _ in range(n_per_quadrant):
            pred = None  # invalid / failed generations count as wrong (strict)
            try:
                midi_path, _ = generate_track(q)
                f = midi_features(midi_path)
                if f is not None:
                    valid += 1
                    pred = clf.predict([f])[0]
            except Exception:
                pass
            y_true.append(q)
            y_pred.append(pred)
    return _score_generation(y_true, y_pred, valid, len(y_true))


def load_midi_pool(midi_dir):
    """Pre-generated MIDI files, grouped by the quadrant they were generated for.

    Quadrant is read from the file name prefix (Q1_000.mid, generated_Q2.mid ...)
    or from a parent folder named Q1..Q4.
    """
    pool = {q: [] for q in QUADRANTS}
    for f in find_midi_files(midi_dir):
        name = Path(f).stem.upper()
        q = next((k for k in QUADRANTS if name.startswith(k) or name.startswith("GENERATED_" + k)), None)
        if q is None:
            q = norm_q(Path(f).parent.name)
        if q in pool:
            pool[q].append(f)
    return pool


def eval_generation_from_pool(clf, pool):
    y_true, y_pred = [], []
    valid = 0
    for q in QUADRANTS:
        for f in sorted(pool[q]):
            feats = midi_features(f)
            pred = None
            if feats is not None:
                valid += 1
                pred = clf.predict([feats])[0]
            y_true.append(q)
            y_pred.append(pred)
    return _score_generation(y_true, y_pred, valid, len(y_true))


# --------------------------------------------------------------------------
# Level 3: end-to-end
# --------------------------------------------------------------------------
def eval_end_to_end(records, clf, generate_track, max_n=40, seed=0):
    """records: list of (ground_truth_quadrant, predicted_quadrant)."""
    if not records:
        return None
    recs = list(records)
    random.Random(seed).shuffle(recs)
    ok = 0
    n = 0
    for gt, pred_q in recs[:max_n]:
        n += 1
        try:
            midi_path, _ = generate_track(pred_q if pred_q in QUADRANTS else "Q4")
            f = midi_features(midi_path)
            if f is not None and clf.predict([f])[0] == gt:
                ok += 1
        except Exception:
            pass
    return {"n": n, "accuracy": ok / n if n else None}


def eval_end_to_end_from_pool(records, clf, pool, max_n=200, seed=0):
    """Same as eval_end_to_end, but each predicted quadrant is answered by a
    random pre-generated clip for that quadrant instead of a fresh generation."""
    if not records:
        return None
    rng = random.Random(seed)
    recs = list(records)
    rng.shuffle(recs)
    ok = n = 0
    for gt, pred_q in recs[:max_n]:
        n += 1
        files = pool.get(pred_q) or []
        if not files:
            continue
        f = midi_features(rng.choice(files))
        if f is not None and clf.predict([f])[0] == gt:
            ok += 1
    return {"n": n, "accuracy": ok / n if n else None}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--face_dir", help="folder with one sub-folder per emotion")
    ap.add_argument("--text_csv", help="CSV with columns text,quadrant")
    ap.add_argument("--emopia_dir", required=True, help="EMOPIA MIDI folder (files named Q1_*.mid ...)")
    ap.add_argument("--midi_dir", help="score pre-generated MIDI instead of calling the generator "
                                        "(files named Q1_*.mid ... or in folders Q1..Q4)")
    ap.add_argument("--n_per_quadrant", type=int, default=25)
    ap.add_argument("--max_per_class", type=int, default=50)
    ap.add_argument("--max_e2e", type=int, default=40)
    ap.add_argument("--out", default="results/evaluation.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    generate_track, pool = None, None
    if args.midi_dir:
        pool = load_midi_pool(args.midi_dir)
        print("Pre-generated clips found:", {q: len(v) for q, v in pool.items()})
    else:
        from src.generator import generate_track

    out = {"generated_at": time.strftime("%Y-%m-%d %H:%M"), "chance_level": 0.25}

    print("Training independent EMOPIA quadrant classifier ...")
    clf, ceiling, n_clips = train_music_classifier(args.emopia_dir, args.seed)
    out["classifier"] = {"real_music_accuracy": ceiling, "n_training_clips": n_clips}

    face_records, text_records = [], []
    if args.face_dir:
        from src.vision import analyze_facial_affect
        print("Evaluating face pipeline ...")
        out["face"], face_records = eval_faces(args.face_dir, analyze_facial_affect, args.max_per_class, args.seed)
    if args.text_csv:
        from src.sentiment import map_text_to_quadrant
        print("Evaluating text pipeline ...")
        out["text"], text_records = eval_text(args.text_csv, map_text_to_quadrant, seed=args.seed)

    print("Evaluating music generator ...")
    if pool is not None:
        out["generation"] = eval_generation_from_pool(clf, pool)
        out["generation_source"] = "pre-generated MIDI files"
    else:
        out["generation"] = eval_generation(clf, generate_track, args.n_per_quadrant)
        out["generation_source"] = "live generator"

    print("Evaluating end-to-end ...")
    e2e = {}
    run_e2e = (lambda recs: eval_end_to_end_from_pool(recs, clf, pool, 200, args.seed)) if pool is not None \
        else (lambda recs: eval_end_to_end(recs, clf, generate_track, args.max_e2e, args.seed))
    if face_records:
        e2e["face"] = run_e2e(face_records)
    if text_records:
        e2e["text"] = run_e2e(text_records)
    out["end_to_end"] = e2e

    # Headline = mean end-to-end accuracy over available input modalities;
    # falls back to generator conditioning accuracy if no perception data given.
    accs = [v["accuracy"] for v in e2e.values() if v and v.get("accuracy") is not None]
    out["headline"] = {
        "accuracy": float(np.mean(accs)) if accs else out["generation"]["conditioning_accuracy"],
        "definition": "end-to-end" if accs else "generator conditioning",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()