import gradio as gr
from src.sentiment import map_text_to_quadrant
from src.vision import analyze_facial_affect
from src.generator import generate_track

def handle_webcam(image):
    if image is None:
        return "⚠️ Please capture a photo or upload an image.", None, None
    try:
        emotion, quadrant, score = analyze_facial_affect(image)
        status = f"📸 **Detected 7-Emotion:** `{emotion.upper()}` ({score:.1%}) → **Target Quadrant:** `{quadrant}`"
        midi_path, wav_path = generate_track(quadrant)
        return status, wav_path, midi_path
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def handle_text(text):
    if not text or not text.strip():
        return "⚠️ Please enter a text prompt.", None, None
    try:
        quadrant, label, score = map_text_to_quadrant(text)
        status = f"✍️ **Text Sentiment:** `{label.upper()}` ({score:.1%}) → **Target Quadrant:** `{quadrant}`"
        midi_path, wav_path = generate_track(quadrant)
        return status, wav_path, midi_path
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

with gr.Blocks(title="Emotion-Conditioned Symbolic Music AI") as demo:
    gr.Markdown("# 🎹 Emotion-Conditioned Symbolic Music AI")
    gr.Markdown("Real-time symbolic piano composition conditioned on facial expressions (7 emotions) or text sentiments.")

    with gr.Tabs():
        with gr.TabItem("📷 Live Webcam / Upload"):
            with gr.Row():
                with gr.Column():
                    webcam_in = gr.Image(sources=["webcam", "upload"], type="numpy", label="Capture / Upload Face")
                    btn_cam = gr.Button("Generate from Expression", variant="primary")
                with gr.Column():
                    status_cam = gr.Markdown("Status: Ready.")
                    audio_cam = gr.Audio(label="Synthesized Piano Audio", type="filepath")
                    file_cam = gr.File(label="Download MIDI (.mid)")

            btn_cam.click(fn=handle_webcam, inputs=[webcam_in], outputs=[status_cam, audio_cam, file_cam])

        with gr.TabItem("✍️ Text Sentiment"):
            with gr.Row():
                with gr.Column():
                    text_in = gr.Textbox(label="Describe your feeling", placeholder="e.g., A quiet and gentle autumn evening...")
                    btn_text = gr.Button("Generate from Sentiment", variant="primary")
                with gr.Column():
                    status_text = gr.Markdown("Status: Ready.")
                    audio_text = gr.Audio(label="Synthesized Piano Audio", type="filepath")
                    file_text = gr.File(label="Download MIDI (.mid)")

            btn_text.click(fn=handle_text, inputs=[text_in], outputs=[status_text, audio_text, file_text])

if __name__ == "__main__":
    demo.launch()
