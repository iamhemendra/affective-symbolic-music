"""Generate a batch of clips per quadrant for evaluation.

Run this where `symusic`/`miditok` load fine (e.g. Google Colab), from the
project root:   python generate_batch.py --n 25 --out generated_eval
Then copy the output folder back and score it locally with:
    python -m src.evaluate --emopia_dir <EMOPIA midi> --midi_dir generated_eval
"""
import argparse
import os
import shutil

from src.generator import generate_track

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=25, help="clips per quadrant")
ap.add_argument("--out", default="generated_eval")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
for q in ["Q1", "Q2", "Q3", "Q4"]:
    for i in range(args.n):
        try:
            midi_path, _ = generate_track(q)  # generator reuses one file name, so copy it out
            shutil.copy(midi_path, os.path.join(args.out, f"{q}_{i:03d}.mid"))
        except Exception as e:
            print(f"{q} #{i} failed: {e}")
    print(f"{q} done")
print("Saved to", args.out)
