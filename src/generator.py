import os
from pathlib import Path
import torch
from miditok import REMI, TokenizerConfig
from transformers import GPT2LMHeadModel
from src.synthesizer import midi_to_wav

config = TokenizerConfig(
    pitch_range=(21, 109),
    beat_res={(0, 4): 8, (4, 12): 4},
    num_velocities=32,
    use_tempos=True,
    use_time_signatures=False,
    use_rests=True,
)
tokenizer = REMI(config)

def generate_track(quadrant: str, max_tokens: int = 150) -> tuple[str, str | None]:
    checkpoint_dir = Path(f"checkpoints/model_{quadrant}")
    if not checkpoint_dir.exists():
        checkpoint_dir = Path(f"model_{quadrant}")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory for {quadrant} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(str(checkpoint_dir)).to(device)
    model.eval()

    prompt = torch.tensor([[tokenizer["BOS_None"]]], device=device)

    with torch.no_grad():
        token_output = model.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.85,
            top_p=0.92,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer["EOS_None"],
        )

    out_tokens = token_output[0].cpu().tolist()
    score = tokenizer.decode([out_tokens])
    
    midi_path = f"generated_{quadrant}.mid"
    wav_path = f"generated_{quadrant}.wav"
    
    score.dump_midi(midi_path)
    synth_success = midi_to_wav(midi_path, wav_path)
    
    return midi_path, (wav_path if synth_success else None)
