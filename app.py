import os
import sys
import tempfile
import asyncio
import gradio as gr

# Ensure MoviePy v1.0.3 import compatibility
try:
    from moviepy.editor import (
        VideoFileClip,
        AudioFileClip,
        TextClip,
        CompositeVideoClip,
        ColorClip,
        ImageClip,
    )
    import moviepy.video.fx.all as vfx
except ModuleNotFoundError:
    # Graceful fallback if moviepy 2.0+ is installed unexpectedly
    from moviepy import (
        VideoFileClip,
        AudioFileClip,
        TextClip,
        CompositeVideoClip,
        ColorClip,
        ImageClip,
    )

import edge_tts

# ==============================================================================
# 1. CORE LOGIC & SYNTHESIS
# ==============================================================================

async def generate_speech_async(text: str, voice: str, output_audio_path: str):
    """Generates TTS audio file using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_audio_path)

def generate_video(script_text, voice, bg_color, text_color, font_size, fps):
    """Generates a styled video with TTS voiceover and captions."""
    if not script_text.strip():
        raise gr.Error("Please enter a text script!")

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "speech.mp3")
    video_path = os.path.join(temp_dir, "output.mp4")

    try:
        # Step 1: Generate TTS Audio
        asyncio.run(generate_speech_async(script_text, voice, audio_path))
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

        # Step 2: Create Background Clip
        bg_clip = ColorClip(
            size=(1080, 1920), color=_hex_to_rgb(bg_color), duration=duration
        )

        # Step 3: Create Text Overlay
        # Note: Uses standard DejaVu-Sans available in Debian Linux Docker images
        try:
            txt_clip = TextClip(
                script_text,
                fontsize=int(font_size),
                color=text_color,
                font="DejaVu-Sans",
                method="caption",
                size=(900, None),
            )
        except Exception:
            # Fallback to default system font if specific font fails
            txt_clip = TextClip(
                script_text,
                fontsize=int(font_size),
                color=text_color,
                method="caption",
                size=(900, None),
            )

        txt_clip = txt_clip.set_position("center").set_duration(duration)

        # Step 4: Composite and Export
        final_video = CompositeVideoClip([bg_clip, txt_clip])
        final_video = final_video.set_audio(audio_clip)

        final_video.write_videofile(
            video_path,
            fps=int(fps),
            codec="libx264",
            audio_codec="aac",
            threads=2,
            preset="ultrafast",
        )

        # Close clips to free memory/locks
        audio_clip.close()
        bg_clip.close()
        txt_clip.close()
        final_video.close()

        return video_path

    except Exception as e:
        raise gr.Error(f"Video Generation Failed: {str(e)}")

def _hex_to_rgb(hex_str):
    """Utility to convert #RRGGBB hex strings to RGB tuples."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


# ==============================================================================
# 2. GRADIO INTERFACE
# ==============================================================================

VOICE_OPTIONS = [
    "en-US-ChristopherNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
]

def create_app():
    with gr.Blocks(title="AI Video Generator") as demo:
        gr.Markdown("# 🎬 Auto Video Generator")
        gr.Markdown("Generate voiceover videos from scripts directly on Railway.")

        with gr.Row():
            with gr.Column():
                script_input = gr.Textbox(
                    label="Script Text",
                    placeholder="Enter the script for your video...",
                    lines=5,
                    value="Welcome to the Auto Video Generator! This video was generated entirely on Railway.",
                )
                voice_select = gr.Dropdown(
                    label="Voice", choices=VOICE_OPTIONS, value="en-US-ChristopherNeural"
                )

                with gr.Accordion("Video Styling Options", open=False):
                    bg_color_picker = gr.ColorPicker(
                        label="Background Color", value="#0f172a"
                    )
                    text_color_picker = gr.ColorPicker(
                        label="Text Color", value="#ffffff"
                    )
                    font_size_slider = gr.Slider(
                        minimum=20, maximum=100, value=50, step=2, label="Font Size"
                    )
                    fps_slider = gr.Slider(
                        minimum=15, maximum=60, value=30, step=5, label="FPS"
                    )

                generate_btn = gr.Button("🚀 Generate Video", variant="primary")

            with gr.Column():
                video_output = gr.Video(label="Generated Output")

        generate_btn.click(
            fn=generate_video,
            inputs=[
                script_input,
                voice_select,
                bg_color_picker,
                text_color_picker,
                font_size_slider,
                fps_slider,
            ],
            outputs=[video_output],
        )

    return demo


# ==============================================================================
# 3. RAILWAY ENTRY POINT & PORT BINDING
# ==============================================================================

if __name__ == "__main__":
    app = create_app()

    # Railway assigns a dynamic PORT via environment variable (defaults to 7860)
    port = int(os.environ.get("PORT", 7860))

    # Server MUST listen on 0.0.0.0 to accept Railway's external proxy traffic
    app.launch(server_name="0.0.0.0", server_port=port)
