# Emotion-Conditioned Symbolic Music Generation

An end-to-end conditional generative sequence framework mapping real-time 7-emotion facial expressions and natural language sentiment to structured polyphonic MIDI piano music across Russell's Affect Quadrants (Q1–Q4).

## System Architecture
- **Vision:** Hugging Face FER image classifier detecting 7 canonical emotions (Happy, Surprise, Angry, Fear, Disgust, Sad, Neutral).
- **NLP:** DistilRoBERTa sentiment classifier mapping text to continuous 2D Valence-Arousal coordinates.
- **Generative Engine:** Causal GPT-2 decoders trained on the EMOPIA dataset using MidiTok's REMI event representation.
- **Synthesis:** Headless FluidSynth rendering pipeline outputting playable WAV files and downloadable standard MIDI files.

## Quickstart

### Prerequisites
- Python 3.10+
- FluidSynth:
  - Ubuntu/Debian: `sudo apt-get install -y fluidsynth fluid-soundfont-gm`
  - macOS: `brew install fluidsynth`
  - Windows: Download FluidSynth binary and add to PATH.

### Install & Run
```bash
pip install -r requirements.txt
python app.py
```

> **Note:** The app requires trained model checkpoints to run. See the [Training](#training) section below before launching the app for the first time.

## Training

The generative models must be trained locally before running the app. Pre-trained checkpoints are not included in this repository due to file size constraints.

### Dataset
This project uses the [EMOPIA dataset](https://zenodo.org/record/5090631) — a publicly available collection of emotion-labelled piano MIDI recordings categorised across Russell's four affect quadrants (Q1: Happy, Q2: Angry, Q3: Sad, Q4: Relaxed). The training script downloads it automatically (~180 MB).

### Dependencies
Install the `requests` library if not already present:
```bash
pip install requests
```

### Train All 4 Quadrants
```bash
python train.py
```

### Train a Single Quadrant
```bash
python train.py --quadrant Q1
```

Checkpoints are saved to `checkpoints/model_Q1/` through `checkpoints/model_Q4/`.

### Hardware Requirements

| Hardware | Estimated Time per Quadrant |
|---|---|
| NVIDIA GPU (fp16 enabled) | ~10–15 minutes |
| CPU only | ~45–60 minutes |

Mixed precision (`fp16`) is enabled by default and requires an NVIDIA GPU. Training automatically falls back to CPU if no GPU is detected.

### PyTorch + CUDA Setup
Ensure you have the CUDA-enabled build of PyTorch installed:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```