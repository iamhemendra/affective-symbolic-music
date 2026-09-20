import os
import shutil
import platform
import urllib.request
import zipfile
from pathlib import Path


# ──────────────────────────────────────────────
# SOUNDFONT — download if not present
# ──────────────────────────────────────────────
SOUNDFONT_PATH = Path("assets/soundfont.sf2")
SOUNDFONT_URL  = (
    "https://github.com/urish/cinto/raw/master/media/GeneralUser%20GS%20MuseScore%20v1.442.sf2"
)

def ensure_soundfont() -> str:
    """Return path to a valid .sf2 soundfont, downloading one if needed."""
    # 1. Local assets folder (highest priority)
    if SOUNDFONT_PATH.exists() and SOUNDFONT_PATH.stat().st_size > 100_000:
        return str(SOUNDFONT_PATH)

    # 2. Linux system soundfonts
    if platform.system() == "Linux":
        for p in [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2",
        ]:
            if os.path.exists(p):
                return p

    # 3. Windows — common FluidSynth install location
    if platform.system() == "Windows":
        win_candidates = [
            r"C:\tools\fluidsynth\share\soundfonts\default.sf2",
            r"C:\Program Files\FluidSynth\share\soundfonts\default.sf2",
            r"C:\FluidSynth\share\soundfonts\default.sf2",
        ]
        for p in win_candidates:
            if os.path.exists(p):
                return p

    # 4. Auto-download a small GM soundfont as fallback
    print("[synthesizer] Soundfont not found — downloading fallback (~30 MB)...")
    SOUNDFONT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(SOUNDFONT_URL, SOUNDFONT_PATH)
        if SOUNDFONT_PATH.exists() and SOUNDFONT_PATH.stat().st_size > 100_000:
            print(f"[synthesizer] Soundfont saved to {SOUNDFONT_PATH}")
            return str(SOUNDFONT_PATH)
    except Exception as e:
        print(f"[synthesizer] Soundfont download failed: {e}")

    return ""


# ──────────────────────────────────────────────
# FLUIDSYNTH BINARY — find on Windows too
# ──────────────────────────────────────────────
def find_fluidsynth() -> str:
    """Return path to fluidsynth binary or empty string if not found."""
    # Standard PATH lookup
    found = shutil.which("fluidsynth")
    if found:
        return found

    # Windows — common install locations
    if platform.system() == "Windows":
        win_paths = [
            r"C:\tools\fluidsynth\bin\fluidsynth.exe",
            r"C:\Program Files\FluidSynth\bin\fluidsynth.exe",
            r"C:\FluidSynth\bin\fluidsynth.exe",
        ]
        for p in win_paths:
            if os.path.exists(p):
                return p

    return ""


# ──────────────────────────────────────────────
# MIDI → WAV using pyfluidsynth (no binary needed)
# ──────────────────────────────────────────────
def midi_to_wav_python(midi_path: str, wav_path: str, soundfont: str) -> bool:
    """Convert MIDI to WAV using the pyfluidsynth Python binding."""
    try:
        import fluidsynth as fs
        import wave, array, struct

        syn = fs.Synth()
        sfid = syn.sfload(soundfont)
        syn.program_select(0, sfid, 0, 0)

        # Read MIDI and render via fluidsynth
        # pyfluidsynth doesn't directly render files, use midi_to_audio workaround
        raise ImportError("pyfluidsynth render not supported this way")
    except Exception:
        return False


# ──────────────────────────────────────────────
# MIDI → WAV using midi2audio (pip install midi2audio)
# ──────────────────────────────────────────────
def midi_to_wav_midi2audio(midi_path: str, wav_path: str, soundfont: str) -> bool:
    """Convert using midi2audio library (wraps FluidSynth internally)."""
    try:
        from midi2audio import FluidSynth as FS2A
        FS2A(sound_font=soundfont).midi_to_audio(midi_path, wav_path)
        return os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
    except Exception as e:
        print(f"[synthesizer] midi2audio failed: {e}")
        return False


# ──────────────────────────────────────────────
# MIDI → WAV using fluidsynth CLI binary
# ──────────────────────────────────────────────
def midi_to_wav_cli(midi_path: str, wav_path: str, soundfont: str, binary: str) -> bool:
    """Convert using fluidsynth command-line binary."""
    # Use platform-safe null redirect
    null = "NUL" if platform.system() == "Windows" else "/dev/null"
    cmd = f'"{binary}" -ni "{soundfont}" "{midi_path}" -F "{wav_path}" -r 44100 > {null} 2>&1'
    exit_code = os.system(cmd)
    return exit_code == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0


# ──────────────────────────────────────────────
# PUBLIC API — tries all methods in order
# ──────────────────────────────────────────────
def midi_to_wav(midi_path: str, wav_path: str) -> bool:
    """
    Convert a MIDI file to WAV. Tries three methods in order:
      1. midi2audio Python library (recommended, no PATH setup needed)
      2. fluidsynth CLI binary
      3. Fails gracefully (MIDI download still works)
    """
    soundfont = ensure_soundfont()
    if not soundfont:
        print("[synthesizer] No soundfont available — WAV synthesis skipped")
        return False

    # Method 1: midi2audio (easiest on Windows — just pip install midi2audio)
    if midi_to_wav_midi2audio(midi_path, wav_path, soundfont):
        return True

    # Method 2: CLI binary
    binary = find_fluidsynth()
    if binary:
        if midi_to_wav_cli(midi_path, wav_path, soundfont, binary):
            return True
        print(f"[synthesizer] fluidsynth CLI failed (binary: {binary})")
    else:
        print("[synthesizer] fluidsynth binary not found in PATH or common locations")

    print("[synthesizer] WAV synthesis failed — MIDI file is still available for download")
    return False