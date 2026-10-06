# Affective Symbolic Music

**Emotion-conditioned symbolic music generation.** Show the system a face or type how you feel, and it composes a short piano piece (MIDI + rendered audio) that matches the emotional quadrant of your input.

The system maps facial expressions and text sentiment onto the four quadrants of Russell's circumplex model of affect, then samples from a quadrant-specific Transformer trained on the EMOPIA dataset.

```
 face photo ──► FER classifier (7 emotions) ─┐
                                              ├─► quadrant (Q1–Q4) ─► GPT-2 (REMI) ─► MIDI ─► FluidSynth ─► WAV
 text ──────► DistilRoBERTa (7 emotions) ────┘
```

---

## How it works

### 1. Emotion perception

| Input | Model | Output |
|---|---|---|
| Face image (upload or webcam snapshot) | [`dima806/facial_emotions_image_detection`](https://huggingface.co/dima806/facial_emotions_image_detection) | One of 7 emotions: happy, surprise, angry, fear, disgust, sad, neutral |
| Text | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) | One of 7 emotions: joy, surprise, anger, fear, disgust, sadness, neutral |

In both cases the top-scoring label is mapped to a quadrant with a fixed lookup table (`src/vision.py`, `src/sentiment.py`). Face predictions with confidence below 0.35 fall back to Q4.

### 2. Emotion → quadrant mapping

| Quadrant | Valence / Arousal | Face labels | Text labels |
|---|---|---|---|
| **Q1** | High / High (excited, joyful) | happy, surprise | joy, surprise |
| **Q2** | Low / High (tense, angry) | angry, fear, disgust | anger, fear, disgust |
| **Q3** | Low / Low (sad, depressed) | sad | sadness |
| **Q4** | High / Low (calm, relaxed) | neutral | neutral |

> The mapping is discrete. Mapping *neutral* to Q4 and *surprise* to Q1 are design assumptions, not ground truth.

### 3. Music generation

- **Data:** [EMOPIA 1.0](https://zenodo.org/record/5090631), about 1,000 emotion-labelled piano MIDI clips, sorted by quadrant from the filename prefix (`Q1_…` to `Q4_…`). `train.py` downloads it automatically.
- **Representation:** [MidiTok](https://github.com/Natooz/MidiTok) **REMI** tokenization (pitch range 21–109, 32 velocity bins, tempo and rest tokens, 8 steps per beat in beats 0–4 and 4 steps per beat in beats 4–12). The tokenizer configuration is defined in code in both `train.py` and `src/generator.py` and must stay identical.
- **Model:** one causal **GPT-2** per quadrant, trained from scratch (6 layers, 512 hidden dimensions, 8 heads, 512-token context, about 20M parameters).
- **Training:** 5 epochs, batch size 4, AdamW (learning rate 3e-4, weight decay 0.01), 100 warmup steps with linear decay, fp16 mixed precision, gradient clipping at 1.0. Sequences are cut into 512-token windows with 50% overlap.
- **Sampling:** generation starts from the BOS token with temperature 0.85 and top-p 0.92, and produces 150 new tokens by default, which is a few seconds of music.
- **Synthesis:** MIDI is rendered to 44.1 kHz WAV through `midi2audio` (FluidSynth), falling back to the FluidSynth command-line binary. A General MIDI soundfont is found automatically (see below).

### 4. Interface

A [Gradio](https://www.gradio.app/) app with three tabs: **Face** (webcam or upload), **Text** and **Model Accuracy**. The Face and Text tabs each return the detected emotion, its confidence, the mapped quadrant, playable audio, and a downloadable MIDI file. If WAV synthesis fails, the MIDI file is still returned. The **Model Accuracy** tab, and a headline badge under the page title, display the results in `results/evaluation.json` (see [Evaluation](#evaluation)).

---

## Quickstart

### Prerequisites

- Python 3.10+
- [FluidSynth](https://www.fluidsynth.org/) installed and on your `PATH` (Windows: the common install folders `C:\tools\fluidsynth`, `C:\Program Files\FluidSynth` and `C:\FluidSynth` are also checked):
  - Ubuntu/Debian: `sudo apt-get install -y fluidsynth fluid-soundfont-gm`
  - macOS: `brew install fluidsynth`
  - Windows: download the FluidSynth binary
- A soundfont: `assets/soundfont.sf2` if present, otherwise a system one (for example FluidR3 GM on Linux). If none is found, a fallback General MIDI soundfont (about 30 MB) is downloaded to `assets/` on first use.
- A CUDA GPU for training (`train.py` requires one). Inference runs on CPU or GPU.

### Install and run

```bash
git clone https://github.com/iamhemendra/affective-symbolic-music.git
cd affective-symbolic-music
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL printed in the terminal.

> Quadrant checkpoints are required at inference time. `src/generator.py` looks in `checkpoints/model_Q1` … `checkpoints/model_Q4` (or `model_Q1` … `model_Q4` in the repo root). If they are missing, train them first.

### Training

```bash
python train.py                 # train all four quadrants
python train.py --quadrant Q1   # train a single quadrant
```

`train.py` downloads and extracts EMOPIA 1.0 into `data/`, tokenizes the MIDI files of each quadrant, trains a GPT-2 from scratch, and writes checkpoints to `checkpoints/model_Q1` … `checkpoints/model_Q4`. A checkpoint is saved every 200 steps and whenever the average training loss for an epoch improves. Hyperparameters are constants at the top of `train.py`.

---

## Evaluation

A generative system has no single accuracy number, so accuracy is measured at three levels.

| Level | Question | Metrics |
|---|---|---|
| 1. Perception | Is the face or text emotion read correctly? | 7-emotion accuracy, macro-F1 and quadrant accuracy (face); quadrant accuracy and macro-F1 (text) |
| 2. Music generation | Does the music match the requested quadrant? | Conditioning accuracy, valence-axis and arousal-axis accuracy, valid-MIDI rate |
| 3. End-to-end | Does input → music → judged emotion equal the true emotion? | End-to-end accuracy (the headline number when levels 1 and 2 are both run) |

**How the music is judged.** An independent Random Forest classifier is trained on real EMOPIA clips, using 11 features: tempo, note density, velocity mean and spread, pitch mean, spread and range, major/minor tendency, polyphony, and inter-onset interval statistics. A generated clip counts as correct when the judge's quadrant equals the requested quadrant. Failed or empty generations count as wrong. Chance level is 25%. The same judge reports valence-axis and arousal-axis agreement from the sign of each quadrant (Q1 +/+, Q2 −/+, Q3 −/−, Q4 +/−).

### Run it

```bash
pip install scikit-learn pretty_midi

# generator only
python -m src.evaluate --emopia_dir data/EMOPIA_1.0/midis --n_per_quadrant 25

# add perception and end-to-end accuracy
python -m src.evaluate --emopia_dir data/EMOPIA_1.0/midis \
    --face_dir <folder with angry/disgust/fear/happy/neutral/sad/surprise sub-folders> \
    --text_csv <csv with columns text,quadrant>
```

Results are written to `results/evaluation.json` and read by the web app. If the generator cannot run on your machine, create clips elsewhere with `python generate_batch.py --n 25 --out generated_eval` and score them with `--midi_dir generated_eval`. In that mode, each predicted quadrant in the end-to-end test is answered by a random pre-generated clip rather than a fresh generation.

### Results (100 generated clips, 25 per quadrant, run of 2026-10-06)

| Metric | Result |
|---|---|
| Music matches the requested quadrant | **70.0%** (chance 25%; 95% interval roughly 60–78%) |
| Arousal axis (high vs low energy) | 91% |
| Valence axis (positive vs negative) | 76% |
| Valid MIDI output | 100% |
| Judge accuracy on held-out real EMOPIA clips | 66.7% |

| Requested quadrant | Judged Q1 | Judged Q2 | Judged Q3 | Judged Q4 | Accuracy |
|---|---|---|---|---|---|
| Q1 | 19 | 5 | 0 | 1 | 76% |
| Q2 | 5 | 20 | 0 | 0 | 80% |
| Q3 | 3 | 3 | 13 | 6 | 52% |
| Q4 | 2 | 0 | 5 | 18 | 72% |

Most errors confuse quadrants that share the same arousal level (Q1/Q2 and Q3/Q4), and Q3 (sad, low energy) is the weakest quadrant. Face and text perception and end-to-end accuracy have not been measured yet; the headline figure above covers music generation only.

---

## Current status and limitations

This is a working prototype, not a validated research result. Please read these before relying on it:

- **Evaluation is partial.** Emotion fidelity of the generated music is now measured (see [Evaluation](#evaluation)), but training still uses all clips with no validation split, and the "best" checkpoint is chosen by training loss. Face and text perception and end-to-end accuracy have not been measured yet, and there is no human listening study.
- **The judge is a proxy.** It is a feature-based classifier trained on the same EMOPIA data as the generators. It reaches only 66.7% on held-out real clips, so a 70% score means the output is about as recognisable as real music to this classifier, not that listeners agree. That 66.7% comes from a clip-level split, so segments of the same song can appear on both sides and the figure may be optimistic.
- **Quadrant models are independent and unconditional.** There is no shared model and there are no control tokens. Each quadrant model is trained on only a few hundred clips.
- **Discrete emotion routing.** Both perception models are reduced to a top label and a lookup table. There is no continuous valence-arousal estimate.
- **Single-frame face input.** The webcam tab captures one image per click, not a continuous stream. There is no face detection or cropping before classification.
- **Short outputs.** Generated pieces are only a few seconds long.
- **Possible clip-level leakage** between segments of the same song if an evaluation split is made at the clip level. Split by song.

## Roadmap

- [ ] Song-level train/validation/test split and held-out perplexity
- [ ] Single conditional model with quadrant control tokens vs. four separate models
- [ ] Objective analysis of generated music per quadrant (note density, pitch range, velocity, tempo, mode)
- [x] Quadrant classifier on generated output to measure emotion fidelity (see [Evaluation](#evaluation))
- [ ] Song-level split for the evaluation judge
- [ ] Measure face and text perception and end-to-end accuracy
- [ ] Listening study with human raters
- [ ] Continuous valence-arousal conditioning
- [ ] Face detection and a real-time webcam loop
- [ ] Longer generation with sliding-window context

## Repository structure

```
.
├── app.py             # Gradio application (Face, Text and Model Accuracy tabs)
├── train.py           # Trains the per-quadrant GPT-2 models
├── generate_batch.py  # Generates clips per quadrant for evaluation
├── results/
│   └── evaluation.json  # Latest evaluation results shown in the app
├── src/
│   ├── __init__.py
│   ├── vision.py      # Facial emotion recognition -> quadrant
│   ├── sentiment.py   # Text emotion classification -> quadrant
│   ├── generator.py   # Quadrant-specific music generation
│   ├── synthesizer.py # MIDI -> WAV rendering (soundfont lookup, midi2audio / FluidSynth CLI)
│   ├── evaluate.py    # Accuracy evaluation (perception, generation, end-to-end)
│   └── accuracy_ui.py # Renders the accuracy dashboard from results/evaluation.json
├── AffectiveMusicGen_Capstone_Presentation (1).pptx
├── requirements.txt
└── README.md
```

## Acknowledgments

- **EMOPIA** dataset for emotion-labelled piano MIDI
- **MidiTok** for REMI tokenization
- **Hugging Face Transformers** and the face and text emotion models linked above
- **FluidSynth** and General MIDI soundfonts
- **Russell (1980)**, *A circumplex model of affect*

## Authors

<!-- Add author names and affiliation here -->

## License

<!-- Add a LICENSE file and state it here, e.g. MIT -->
