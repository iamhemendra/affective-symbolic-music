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
