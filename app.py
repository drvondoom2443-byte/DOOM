import os
import sys
import tempfile
import asyncio
import gradio as gr
from PIL import Image

# Ensure MoviePy v1.0.3 compatibility
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
# 1. CORE PROCESSING FUNCTIONS
# ==============================================================================

async def generate_speech_async(text: str, voice: str, output_audio_path: str):
    """Generates TTS audio file using edge-tts."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_audio_path)

def apply_motion_effect(clip, effect_type, duration):
    """Applies camera motion zoom effects to background images."""
    if effect_type == "Zoom In":
        # Gradual zoom from 100% to 115% scale
        return clip.resize(lambda t: 1 + 0.15 * (t / duration))
    elif effect_type == "Zoom Out":
        # Gradual zoom out from 115% to 100% scale
        return clip.resize(lambda t: 1.15 - 0.15 * (t / duration))
    return clip  # Static

def generate_video(
    script_text,
    voice,
    bg_image,
    motion_effect,
    overlay_opacity,
    bg_color,
    text_color,
    text_position,
    font_size,
    fps
):
    if not script_text.strip():
        raise gr.Error("Please enter a script text before generating!")

    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "speech.mp3")
    video_path = os.path.join(temp_dir, "output.mp4")

    try:
        # Step 1: Generate Speech Audio
        asyncio.run(generate_speech_async(script_text, voice, audio_path))
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration

        # Standard vertical Short/Reel resolution (1080x1920)
        target_size = (1080, 1920)

        # Step 2: Prepare Background (Image or Solid Color)
        clips_to_composite = []

        if bg_image is not None:
            # Process uploaded background image
            raw_img = Image.open(bg_image.name if hasattr(bg_image, 'name') else bg_image)
            img_clip = ImageClip(bg_image.name if hasattr(bg_image, 'name') else bg_image)
            
            # Crop/Resize to fit 1080x1920 keeping aspect ratio
            img_clip = img_clip.resize(height=1920)
            if img_clip.w < 1080:
                img_clip = img_clip.resize(width=1080)
                
            img_clip = img_clip.crop(
                x_center=img_clip.w / 2,
                y_center=img_clip.h / 2,
                width=1080,
                height=1920
            ).set_duration(duration)

            # Apply Motion Effect
            img_clip = apply_motion_effect(img_clip, motion_effect, duration)
            clips_to_composite.append(img_clip)

            # Optional Dark Overlay for caption readability
            if overlay_opacity > 0:
                dark_overlay = (
                    ColorClip(size=target_size, color=(0, 0, 0))
                    .set_opacity(overlay_opacity)
                    .set_duration(duration)
                )
                clips_to_composite.append(dark_overlay)
        else:
            # Fallback to Solid Color Background
            bg_clip = ColorClip(
                size=target_size, color=_hex_to_rgb(bg_color), duration=duration
            )
            clips_to_composite.append(bg_clip)

        # Step 3: Text / Captions Rendering
        try:
            txt_clip = TextClip(
                script_text,
                fontsize=int(font_size),
                color=text_color,
                font="DejaVu-Sans",
                method="caption",
                size=(920, None),
            )
        except Exception:
            txt_clip = TextClip(
                script_text,
                fontsize=int(font_size),
                color=text_color,
                method="caption",
                size=(920, None),
            )

        pos_mapping = {
            "Top": ("center", 250),
            "Center": "center",
            "Bottom": ("center", 1500)
        }
        
        txt_clip = txt_clip.set_position(pos_mapping.get(text_position, "center")).set_duration(duration)
        clips_to_composite.append(txt_clip)

        # Step 4: Final Composite & Render
        final_video = CompositeVideoClip(clips_to_composite, size=target_size)
        final_video = final_video.set_audio(audio_clip)

        final_video.write_videofile(
            video_path,
            fps=int(fps),
            codec="libx264",
            audio_codec="aac",
            threads=2,
            preset="ultrafast",
        )

        # Clean up open references
        audio_clip.close()
        final_video.close()

        return video_path

    except Exception as e:
        raise gr.Error(f"Video Generation Failed: {str(e)}")

def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))


# ==============================================================================
# 2. FULL CUSTOM GRADIO INTERFACE
# ==============================================================================

VOICE_OPTIONS = [
    "en-US-ChristopherNeural",
    "en-US-JennyNeural",
    "en-US-GuyNeural",
    "en-GB-SoniaNeural",
]

def create_app():
    with gr.Blocks(title="Auto Video Generator") as demo:
        gr.Markdown("# 🎬 Advanced Auto Video Generator")
        gr.Markdown("Upload images, customize typography, add camera motion, and synthesize speech.")

        with gr.Row():
            # Left Panel - All Customization Inputs
            with gr.Column(scale=1):
                script_input = gr.Textbox(
                    label="Script Text",
                    placeholder="Enter the voiceover script...",
                    lines=4,
                    value="Welcome! You can now upload background images, choose motion effects, and customize caption positions.",
                )
                
                voice_select = gr.Dropdown(
                    label="Voice Model", choices=VOICE_OPTIONS, value="en-US-ChristopherNeural"
                )

                # Background & Image Controls
                with gr.Group():
                    gr.Markdown("### 🖼️ Background & Visuals")
                    bg_image_input = gr.Image(
                        label="Background Image (Optional)", 
                        type="filepath"
                    )
                    motion_select = gr.Radio(
                        label="Camera Motion",
                        choices=["Static", "Zoom In", "Zoom Out"],
                        value="Zoom In"
                    )
                    overlay_slider = gr.Slider(
                        minimum=0.0, maximum=0.8, value=0.3, step=0.05,
                        label="Dark Overlay Tint (Improves Text Readability)"
                    )
                    bg_color_picker = gr.ColorPicker(
                        label="Fallback Background Color", value="#0f172a"
                    )

                # Typography & Formatting
                with gr.Accordion("🎨 Text & Styling Options", open=True):
                    text_color_picker = gr.ColorPicker(
                        label="Text Color", value="#ffffff"
                    )
                    text_pos_select = gr.Radio(
                        label="Text Position",
                        choices=["Top", "Center", "Bottom"],
                        value="Center"
                    )
                    font_size_slider = gr.Slider(
                        minimum=20, maximum=100, value=52, step=2, label="Font Size"
                    )
                    fps_slider = gr.Slider(
                        minimum=15, maximum=60, value=30, step=5, label="FPS"
                    )

                generate_btn = gr.Button("🚀 Generate Video", variant="primary", size="lg")

            # Right Panel - Output Player
            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Output Video")

        generate_btn.click(
            fn=generate_video,
            inputs=[
                script_input,
                voice_select,
                bg_image_input,
                motion_select,
                overlay_slider,
                bg_color_picker,
                text_color_picker,
                text_pos_select,
                font_size_slider,
                fps_slider,
            ],
            outputs=[video_output],
        )

    return demo


# ==============================================================================
# 3. RAILWAY RUNTIME BINDING
# ==============================================================================

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
