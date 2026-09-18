"""
train.py — Fine-tunes GPT-2 on EMOPIA MIDI dataset for Q1–Q4 emotion quadrants.
Saves checkpoints to checkpoints/model_Q1 ... checkpoints/model_Q4

Usage:
    python train.py                  # train all 4 quadrants
    python train.py --quadrant Q1    # train one specific quadrant

Requirements (already in your venv):
    torch, transformers, miditok, requests, zipfile
"""

import os
import sys
import zipfile
import argparse
import requests
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from miditok import REMI, TokenizerConfig
from symusic import Score

# ──────────────────────────────────────────────
# CONFIG — tweak these if needed
# ──────────────────────────────────────────────
DATASET_URL = "https://zenodo.org/record/5090631/files/EMOPIA_1.0.zip"
DATASET_DIR = Path("data/EMOPIA_1.0")
MIDI_DIR    = DATASET_DIR / "midis"
CHECKPOINT_DIR = Path("checkpoints")

EPOCHS      = 5        # increase to 8–10 for better quality
BATCH_SIZE  = 4        # safe for 6 GB VRAM with fp16
MAX_SEQ_LEN = 512      # token sequence length
LR          = 3e-4
WARMUP_STEPS = 100
SAVE_STEPS  = 200      # save checkpoint every N steps
USE_FP16    = True     # mixed precision — keep True for RTX 3050

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]

# ──────────────────────────────────────────────
# TOKENIZER — must match generator.py exactly
# ──────────────────────────────────────────────
config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    use_tempos=True,
    use_time_signatures=False,
    use_rests=True,
)
tokenizer = REMI(config)


# ──────────────────────────────────────────────
# STEP 1 — Download EMOPIA dataset
# ──────────────────────────────────────────────
def download_emopia():
    zip_path = Path("data/EMOPIA_1.0.zip")
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if MIDI_DIR.exists() and any(MIDI_DIR.glob("*.mid")):
        print(f"[v] EMOPIA already downloaded at {MIDI_DIR}")
        return

    print(f"[>] Downloading EMOPIA dataset (~180 MB)...")
    print(f"    URL: {DATASET_URL}")

    response = requests.get(DATASET_URL, stream=True)
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r    {pct:.1f}% ({downloaded/1e6:.1f} MB / {total/1e6:.1f} MB)", end="")
    print()

    print("[v] Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("data/")

    # Find where MIDIs landed after extraction
    possible = list(Path("data").rglob("*.mid"))
    if not possible:
        raise RuntimeError("No .mid files found after extraction. Check the download.")

    midi_parent = possible[0].parent
    if midi_parent != MIDI_DIR:
        print(f"    Moving MIDIs from {midi_parent} -> {MIDI_DIR}")
        MIDI_DIR.mkdir(parents=True, exist_ok=True)
        for f in midi_parent.glob("*.mid"):
            shutil.copy(f, MIDI_DIR / f.name)

    zip_path.unlink()  # free disk space
    print(f"[v] EMOPIA ready — {len(list(MIDI_DIR.glob('*.mid')))} MIDI files")


# ──────────────────────────────────────────────
# STEP 2 — Split MIDIs by quadrant
# EMOPIA filenames: Q1_0001.mid, Q2_0045.mid etc.
# ──────────────────────────────────────────────
def get_midi_files(quadrant: str) -> list:
    files = list(MIDI_DIR.glob(f"{quadrant}_*.mid"))
    if not files:
        files = [f for f in MIDI_DIR.glob("*.mid")
                 if f.name.upper().startswith(quadrant)]
    if not files:
        raise FileNotFoundError(
            f"No MIDI files found for {quadrant} in {MIDI_DIR}.\n"
            f"Sample files: {[f.name for f in MIDI_DIR.glob('*.mid')][:5]}"
        )
    print(f"[v] {quadrant}: {len(files)} MIDI files found")
    return files


# ──────────────────────────────────────────────
# STEP 3 — Tokenize MIDIs
# ──────────────────────────────────────────────
def tokenize_files(midi_files: list) -> list:
    all_sequences = []
    skipped = 0
    for path in midi_files:
        try:
            score = Score(str(path))
            tokens = tokenizer.encode(score)
            for seq in tokens:
                ids = seq.ids
                if len(ids) > 10:
                    all_sequences.append(ids)
        except Exception as e:
            skipped += 1
    if skipped:
        print(f"    Skipped {skipped} files (corrupt/unreadable)")
    print(f"    Tokenized -> {len(all_sequences)} sequences")
    return all_sequences


class MIDIDataset(Dataset):
    def __init__(self, sequences: list, max_len: int):
        self.examples = []
        bos = tokenizer["BOS_None"]
        eos = tokenizer["EOS_None"]
        pad = tokenizer.pad_token_id

        for seq in sequences:
            ids = [bos] + seq + [eos]
            # Sliding window chunks with 50% overlap
            for i in range(0, len(ids), max_len // 2):
                chunk = ids[i : i + max_len]
                if len(chunk) < 16:
                    continue
                padded = chunk + [pad] * (max_len - len(chunk))
                self.examples.append(padded)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.examples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


# ──────────────────────────────────────────────
# STEP 4 — Build GPT-2 model
# ──────────────────────────────────────────────
def build_model(vocab_size: int) -> GPT2LMHeadModel:
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=MAX_SEQ_LEN,
        n_embd=512,      # smaller than default 768 — fits in 6 GB VRAM
        n_layer=6,       # default is 12 — halved for speed/memory
        n_head=8,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    model = GPT2LMHeadModel(gpt2_config)
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Model: {params:.1f}M parameters")
    return model


# ──────────────────────────────────────────────
# STEP 5 — Training loop
# ──────────────────────────────────────────────
def train_quadrant(quadrant: str):
    print(f"\n{'='*55}")
    print(f"  Training quadrant: {quadrant}")
    print(f"{'='*55}")

    save_dir = CHECKPOINT_DIR / f"model_{quadrant}"
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=USE_FP16)

    midi_files = get_midi_files(quadrant)
    sequences  = tokenize_files(midi_files)

    if not sequences:
        print(f"[!] No valid sequences for {quadrant} — skipping")
        return

    dataset = MIDIDataset(sequences, MAX_SEQ_LEN)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=0,   # Windows: must be 0
    )
    print(f"    Dataset: {len(dataset)} chunks | {len(loader)} batches/epoch")

    model     = build_model(len(tokenizer)).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps,
    )

    global_step = 0
    best_loss   = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                outputs = model(input_ids=input_ids, labels=labels)
                loss    = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % 20 == 0:
                avg = epoch_loss / (batch_idx + 1)
                print(f"    Epoch {epoch}/{EPOCHS} | Step {global_step} | "
                      f"Loss {avg:.4f} | LR {scheduler.get_last_lr()[0]:.2e}")

            if global_step % SAVE_STEPS == 0:
                model.save_pretrained(str(save_dir))
                tokenizer.save_pretrained(str(save_dir))
                print(f"    [saved checkpoint at step {global_step}]")

        avg_epoch_loss = epoch_loss / len(loader)
        print(f"\n  Epoch {epoch} done | Avg Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(str(save_dir))
            tokenizer.save_pretrained(str(save_dir))
            print(f"  Best model saved -> {save_dir}")

    print(f"\n[v] {quadrant} training complete! Checkpoint: {save_dir}")
    del model
    torch.cuda.empty_cache()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on EMOPIA per quadrant")
    parser.add_argument(
        "--quadrant",
        choices=QUADRANTS,
        default=None,
        help="Train a single quadrant (default: all 4)",
    )
    args = parser.parse_args()

    print("\n Piano GPT Training — Emotion Quadrants")
    print(f"   Device  : {torch.cuda.get_device_name(0)}")
    print(f"   VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   fp16    : {USE_FP16}")
    print(f"   Epochs  : {EPOCHS}")
    print(f"   Batch   : {BATCH_SIZE}")
    print()

    download_emopia()

    targets = [args.quadrant] if args.quadrant else QUADRANTS
    for q in targets:
        train_quadrant(q)

    print("\n All training complete!")
    print(f"   Checkpoints: {CHECKPOINT_DIR.resolve()}")
    print("   Run `python app.py` to start the app.")


if __name__ == "__main__":
    main()
