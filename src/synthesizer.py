import os
import shutil
import platform

def get_soundfont_path() -> str:
    if os.path.exists("assets/soundfont.sf2"):
        return "assets/soundfont.sf2"

    if platform.system() == "Linux":
        candidates = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/sounds/sf2/default-GM.sf2"
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
    return ""

def midi_to_wav(midi_path: str, wav_path: str) -> bool:
    soundfont = get_soundfont_path()
    if not soundfont or not shutil.which("fluidsynth"):
        return False

    cmd = f'fluidsynth -ni "{soundfont}" "{midi_path}" -F "{wav_path}" -r 44100 > /dev/null 2>&1'
    exit_code = os.system(cmd)
    return exit_code == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 0
