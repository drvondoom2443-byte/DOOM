"""
Auto Video Generator - Gradio app configured for deployment on Railway.

Flow:
  1. Voice: upload your own audio, OR type a script and generate an AI voice (edge-tts).
  2. Images: upload one or more images OR use procedural Nature Scenes.
  3. Effect: pick from looks (Ken Burns, Pan Drift, Camera Shake, Film Frame, Rain).
  4. Captions: burn script text on screen in various styles and Google Fonts.
  5. Generate: renders an MP4 video for download/preview.
"""

import asyncio
import json
import math
import os
import random
import re
import tempfile
import textwrap
import uuid

import edge_tts
import gradio as gr
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from proglog import ProgressBarLogger
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    CompositeVideoClip,
    concatenate_videoclips,
)

OUT_DIR = os.path.join(tempfile.gettempdir(), "auto_video_gen")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_DIMENSION = 854
MAX_SCRIPT_CHARS = 500_000
WARN_AUDIO_SECONDS = 120
MAX_AUDIO_SECONDS = 3600

FAVORITES_PATH = os.path.join(OUT_DIR, "favorite_voices.json")

LOCALE_NAMES = {
    "en-US": "English (US)", "en-GB": "English (UK)", "en-IN": "English (India)",
    "en-AU": "English (Australia)", "en-CA": "English (Canada)", "en-IE": "English (Ireland)",
    "en-ZA": "English (South Africa)", "en-NG": "English (Nigeria)", "en-PH": "English (Philippines)",
    "ur-PK": "Urdu (Pakistan)", "ur-IN": "Urdu (India)",
    "hi-IN": "Hindi (India)", "ar-SA": "Arabic (Saudi Arabia)", "ar-EG": "Arabic (Egypt)",
    "ru-RU": "Russian", "de-DE": "German", "fr-FR": "French", "fr-CA": "French (Canada)",
    "es-ES": "Spanish (Spain)", "es-MX": "Spanish (Mexico)", "es-US": "Spanish (US)",
    "pt-BR": "Portuguese (Brazil)", "pt-PT": "Portuguese (Portugal)",
    "it-IT": "Italian", "tr-TR": "Turkish", "nl-NL": "Dutch", "pl-PL": "Polish",
    "ja-JP": "Japanese", "ko-KR": "Korean", "zh-CN": "Chinese (Mandarin, Simplified)",
    "zh-TW": "Chinese (Taiwan)", "zh-HK": "Chinese (Cantonese, HK)",
    "id-ID": "Indonesian", "vi-VN": "Vietnamese", "th-TH": "Thai", "bn-IN": "Bengali (India)",
    "bn-BD": "Bengali (Bangladesh)", "fa-IR": "Persian", "pa-IN": "Punjabi (India)",
    "sw-KE": "Swahili (Kenya)", "uk-UA": "Ukrainian", "ro-RO": "Romanian", "el-GR": "Greek",
    "he-IL": "Hebrew", "sv-SE": "Swedish", "no-NO": "Norwegian", "da-DK": "Danish", "fi-FI": "Finnish",
}

ALL_VOICES_CACHE_PATH = os.path.join(OUT_DIR, "edge_voices_cache.json")
_all_voices_memory_cache = None


def _locale_label(locale):
    return LOCALE_NAMES.get(locale, locale)


def _fetch_all_edge_voices():
    global _all_voices_memory_cache
    if _all_voices_memory_cache:
        return _all_voices_memory_cache

    if os.path.exists(ALL_VOICES_CACHE_PATH):
        try:
            with open(ALL_VOICES_CACHE_PATH, "r", encoding="utf-8") as f:
                _all_voices_memory_cache = json.load(f)
                if _all_voices_memory_cache:
                    return _all_voices_memory_cache
        except Exception:
            pass

    async def _list():
        return await edge_tts.list_voices()

    try:
        raw = asyncio.run(_list())
    except Exception:
        raw = []

    voices = []
    for v in raw:
        short_name = v.get("ShortName")
        locale = v.get("Locale")
        gender = v.get("Gender")
        if not short_name or not locale:
            continue
        friendly = short_name.split("-")[-1].replace("Neural", "")
        voices.append({
            "ShortName": short_name,
            "Locale": locale,
            "Gender": gender or "Unknown",
            "FriendlyName": friendly,
        })
    voices.sort(key=lambda v: (v["Locale"], v["FriendlyName"]))

    if voices:
        _all_voices_memory_cache = voices
        try:
            with open(ALL_VOICES_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(voices, f)
        except Exception:
            pass
    return voices


def _all_locales():
    voices = _fetch_all_edge_voices()
    return sorted(set(v["Locale"] for v in voices))


def _voice_display_label(v):
    return f"{v['FriendlyName']} - {v['Gender']} ({_locale_label(v['Locale'])}) [{v['ShortName']}]"


def _load_favorites():
    if os.path.exists(FAVORITES_PATH):
        try:
            with open(FAVORITES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _filtered_voice_choices(language_choice, gender_choice, favorites_only):
    voices = _fetch_all_edge_voices()
    favorites = set(_load_favorites())
    out = []
    for v in voices:
        if language_choice and language_choice != "All Languages" and _locale_label(v["Locale"]) != language_choice:
            continue
        if gender_choice and gender_choice != "All" and v["Gender"] != gender_choice:
            continue
        if favorites_only and v["ShortName"] not in favorites:
            continue
        out.append(_voice_display_label(v))
    return out if out else ["Microsoft Server Speech Text to Speech Voice (en-US, JennyNeural) [en-US-JennyNeural]"]


def _shortname_from_label(label):
    m = re.search(r"\[([^\]]+)\]$", label or "")
    return m.group(1) if m else "en-US-JennyNeural"


EFFECT_OPTIONS = {
    "None (static)": "none",
    "Zoom (Ken Burns)": "kenburns",
    "Pan Drift": "pan",
    "Camera Shake": "shake",
    "Film Frame (vintage)": "filmframe",
    "Rain Overlay": "rain",
}

CAPTION_STYLE_OPTIONS = {
    "Classic Bottom": "classic",
    "Bold Outline": "outline",
    "Boxed": "boxed",
    "Minimal Top": "minimal",
}

RENDER_SPEED_OPTIONS = {
    "Fast (quick render, lower quality)": (12, 640),
    "Balanced (recommended)": (16, 854),
    "High Quality (slow)": (24, 1280),
}

ASPECT_RATIO_OPTIONS = {
    "16:9 (YouTube Landscape)": (16, 9),
    "9:16 (Shorts / Reels / TikTok)": (9, 16),
    "1:1 (Square)": (1, 1),
}


def _resolve_dimensions(speed_choice, aspect_choice):
    fps, long_side = RENDER_SPEED_OPTIONS.get(speed_choice, RENDER_SPEED_OPTIONS["Balanced (recommended)"])
    rw, rh = ASPECT_RATIO_OPTIONS.get(aspect_choice, ASPECT_RATIO_OPTIONS["16:9 (YouTube Landscape)"])
    if rw >= rh:
        width = long_side
        height = int(round(long_side * rh / rw))
    else:
        height = long_side
        width = int(round(long_side * rw / rh))
    width -= width % 2
    height -= height % 2
    return fps, max(width, 2), max(height, 2)


# Procedural Nature Generators
def _vertical_gradient(w, h, top_rgb, bottom_rgb):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top_rgb[0] + (bottom_rgb[0] - top_rgb[0]) * t)
        g = int(top_rgb[1] + (bottom_rgb[1] - top_rgb[1]) * t)
        b = int(top_rgb[2] + (bottom_rgb[2] - top_rgb[2]) * t)
        for x in range(0, w, 4):
            px[x, y] = (r, g, b)
    return img.resize((w, h))


def _generate_sky_scene(w, h):
    img = _vertical_gradient(w, h, (94, 170, 235), (214, 238, 252))
    draw = ImageDraw.Draw(img, "RGBA")
    sun_x, sun_r = random.uniform(0.62, 0.86), int(h * random.uniform(0.07, 0.11))
    draw.ellipse([w * sun_x - sun_r, h * 0.16 - sun_r, w * sun_x + sun_r, h * 0.16 + sun_r], fill=(255, 250, 214, 235))
    for _ in range(random.randint(3, 6)):
        cx, cy, s = random.uniform(0.05, 0.95), random.uniform(0.1, 0.45), random.uniform(0.5, 1.1)
        cw, ch = int(w * 0.16 * s), int(h * 0.05 * s)
        x, y = int(w * cx), int(h * cy)
        for dx, dy, sc in [(-cw * 0.5, 0, 0.7), (0, -ch * 0.3, 1.0), (cw * 0.5, 0, 0.75)]:
            draw.ellipse([x + dx - cw * sc / 2, y + dy - ch * sc / 2, x + dx + cw * sc / 2, y + dy + ch * sc / 2], fill=(255, 255, 255, 210))
    return img


def _draw_mountain_layers(draw, w, h, base_top, base_bottom, colors, jitter=(0.06, 0.22)):
    n = len(colors)
    for i, color in enumerate(colors):
        base_y = h * (base_top + (base_bottom - base_top) * (i + 1) / n)
        pts, x, step = [(0, h)], 0, w / random.randint(6, 9)
        while x <= w + step:
            pts.append((x, base_y - random.uniform(*jitter) * h))
            x += step
        pts.append((w, h))
        draw.polygon(pts, fill=color)


def _generate_mountain_scene(w, h):
    img = _vertical_gradient(w, h, (255, 200, 150), (255, 236, 214))
    draw = ImageDraw.Draw(img, "RGBA")
    sun_r = int(h * 0.07)
    draw.ellipse([w * 0.5 - sun_r, h * 0.32 - sun_r, w * 0.5 + sun_r, h * 0.32 + sun_r], fill=(255, 244, 200, 255))
    _draw_mountain_layers(draw, w, h, 0.45, 1.0, [(150, 160, 190), (110, 122, 158), (70, 82, 120), (38, 46, 74)])
    return img


def _generate_river_scene(w, h):
    img = _vertical_gradient(w, h, (150, 205, 240), (225, 245, 250))
    draw = ImageDraw.Draw(img, "RGBA")
    bank_h = int(h * 0.16)
    draw.rectangle([0, 0, w, bank_h], fill=(120, 170, 90))
    draw.rectangle([0, h - bank_h, w, h], fill=(96, 148, 74))
    river = Image.new("RGB", (w, max(1, h - 2 * bank_h)), (72, 138, 178))
    img.paste(river, (0, bank_h))
    return img


NATURE_SCENES = {
    "Sky": _generate_sky_scene,
    "Mountains": _generate_mountain_scene,
    "River": _generate_river_scene,
}


class GradioRenderLogger(ProgressBarLogger):
    def __init__(self, gr_progress, start=0.7, end=0.98):
        super().__init__()
        self.gr_progress = gr_progress
        self.start = start
        self.end = end

    def bars_callback(self, bar, attr, value, old_value=None):
        if attr == "index":
            total = self.bars[bar].get("total")
            if total:
                frac = value / total
                overall = self.start + frac * (self.end - self.start)
                self.gr_progress(min(overall, self.end), desc="Rendering video...")


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
]

FONT_CACHE_DIR = os.path.join(OUT_DIR, "fonts")
os.makedirs(FONT_CACHE_DIR, exist_ok=True)

GOOGLE_FONTS = {
    "Default (built-in)": None,
    "Poppins": "Poppins",
    "Roboto": "Roboto",
    "Montserrat": "Montserrat",
    "Oswald": "Oswald",
}


def _download_google_font(font_family, weight=700):
    if not font_family:
        return None
    cache_path = os.path.join(FONT_CACHE_DIR, f"{font_family.replace(' ', '_')}_{weight}.ttf")
    if os.path.exists(cache_path):
        return cache_path
    try:
        css_url = f"https://fonts.googleapis.com/css2?family={font_family.replace(' ', '+')}:wght@{weight}&display=swap"
        headers = {"User-Agent": "Mozilla/4.0 (compatible; MSIE 5.0; Windows 98)"}
        resp = requests.get(css_url, headers=headers, timeout=10)
        match = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)", resp.text)
        if not match:
            return None
        ttf_bytes = requests.get(match.group(1), timeout=10).content
        with open(cache_path, "wb") as f:
            f.write(ttf_bytes)
        return cache_path
    except Exception:
        return None


def _hex_to_rgba(hex_color, alpha=255):
    if not hex_color:
        return (255, 255, 255, alpha)
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255, alpha)
    return (int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16), alpha)


def _load_font(size, font_path=None):
    if font_path and os.path.exists(font_path):
        try:
            return ImageFont.truetype(font_path, size)
        except Exception:
            pass
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _cover_resize(img, target_w, target_h):
    img = img.convert("RGB")
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale) + 1, int(src_h * scale) + 1
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - target_w) // 2, (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _apply_sepia(np_img):
    img = np_img.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    tr = 0.393 * r + 0.769 * g + 0.189 * b
    tg = 0.349 * r + 0.686 * g + 0.168 * b
    tb = 0.272 * r + 0.534 * g + 0.131 * b
    return np.clip(np.stack([tr, tg, tb], axis=-1), 0, 255).astype(np.uint8)


def _make_vignette_mask(h, w):
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
    return np.clip(1.0 - (dist - 0.45) * 0.85, 0.35, 1.0)


def _apply_vignette(np_img, mask):
    return np.clip(np_img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def _draw_film_bars(pil_img):
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    bar_h = max(6, int(h * 0.055))
    draw.rectangle([(0, 0), (w, bar_h)], fill=(0, 0, 0))
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0))
    return pil_img


def _make_caption_frame(text, width, height, style="classic", font_path=None, custom_color=None,
                         border_on=None, text_case="Normal"):
    if text_case == "UPPERCASE":
        text = text.upper()
    elif text_case == "lowercase":
        text = text.lower()

    max_block_h = int(height * (0.14 if style == "minimal" else 0.20))
    size_divisor = {"classic": 38, "outline": 28, "boxed": 34, "minimal": 44}.get(style, 38)
    font_size = max(13, min(width // size_divisor, height // 13))

    for _ in range(4):
        font = _load_font(font_size, font_path)
        wrapped = textwrap.fill(text, width=max(10, width // max(1, font_size // 2)))
        lines = wrapped.split("\n")
        line_h = int(font_size * 1.3)
        block_h = line_h * len(lines) + 40
        if block_h <= max_block_h or font_size <= 14:
            break
        font_size = max(14, int(font_size * (max_block_h / block_h)))

    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)

    y = int(height * 0.06) if style == "minimal" else height - block_h - 30

    outline_fill, outline_w = None, 0
    if style == "classic":
        draw.rectangle([(0, y - 20), (width, y + block_h)], fill=(0, 0, 0, 140))
        text_fill, outline_fill, outline_w = (255, 255, 255, 255), (0, 0, 0, 255), 2
    elif style == "outline":
        text_fill = (255, 210, 63, 255)
        outline_fill, outline_w = (0, 0, 0, 255), max(3, font_size // 10)
    elif style == "boxed":
        widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines) if lines else 100
        box_x0 = (width - widest) // 2 - 24
        box_x1 = (width + widest) // 2 + 24
        draw.rounded_rectangle([(box_x0, y - 18), (box_x1, y + block_h - 4)], radius=14, fill=(0, 0, 0, 160))
        text_fill = (255, 255, 255, 255)
    else:
        text_fill, outline_fill, outline_w = (242, 239, 233, 255), (0, 0, 0, 160), 1

    if custom_color:
        text_fill = _hex_to_rgba(custom_color)
    if border_on is not None:
        if border_on and not outline_fill:
            outline_fill, outline_w = (0, 0, 0, 255), max(2, font_size // 12)
        elif not border_on:
            outline_fill, outline_w = None, 0

    yy = y if style == "minimal" else y + 20
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (width - line_w) // 2
        if outline_fill and outline_w > 0:
            step = max(1, outline_w)
            for dx in range(-outline_w, outline_w + 1, step):
                for dy in range(-outline_w, outline_w + 1, step):
                    if dx or dy:
                        draw.text((x + dx, yy + dy), line, font=font, fill=outline_fill)
        draw.text((x, yy), line, font=font, fill=text_fill)
        yy += line_h

    return np.array(frame)


def _make_image_clip(path_or_img, width, height, duration, effects, seed=0):
    if isinstance(path_or_img, str):
        img = Image.open(path_or_img)
    else:
        img = path_or_img

    effects = [e for e in (effects or []) if e and e != "none"]

    if not effects:
        pil_img = _cover_resize(img, width, height)
        return ImageClip(np.array(pil_img)).set_duration(duration)

    has_kb, has_pan = "kenburns" in effects, "pan" in effects
    has_shake, has_film = "shake" in effects, "filmframe" in effects

    zoom_margin = 1.0 + (0.16 if has_kb else 0) + (0.18 if has_pan else 0) + (0.08 if has_shake else 0)
    big_w, big_h = int(width * zoom_margin), int(height * zoom_margin)
    big_img = _cover_resize(img, big_w, big_h)

    pan_direction = 1 if seed % 2 == 0 else -1
    pan_room_x = (big_w - width) * 0.35
    shake_room_x, shake_room_y = (big_w - width) * 0.22, (big_h - height) * 0.22

    vignette_mask = _make_vignette_mask(height, width) if has_film else None
    bars_alpha = bars_rgb = grain_pool = None
    if has_film:
        bars_rgba = np.array(_draw_film_bars(Image.new("RGBA", (width, height), (0, 0, 0, 0))))
        bars_alpha = (bars_rgba[..., 3:4].astype(np.float32)) / 255.0
        bars_rgb = bars_rgba[..., :3].astype(np.float32)
        grain_pool = [np.random.randint(-5, 6, (height, width, 1)).astype(np.int16) for _ in range(8)]

    def make_frame(t):
        progress = min(1.0, t / duration) if duration > 0 else 0.0
        zoom_amt = (0.16 if has_kb else 0.0) + (0.06 if has_film else 0.0)
        scale = (1 + zoom_amt) - zoom_amt * progress
        cw, ch = width * scale, height * scale

        dx = dy = 0.0
        if has_pan:
            dx += pan_direction * (progress - 0.5) * pan_room_x * 2
        if has_shake:
            dx += math.sin(t * 1.6 + seed * 3) * shake_room_x * 0.7
            dy += math.cos(t * 1.9 + seed * 2) * shake_room_y * 0.7

        left = max(0, min(big_w - cw, (big_w - cw) / 2 + dx))
        top = max(0, min(big_h - ch, (big_h - ch) / 2 + dy))
        cropped = big_img.crop((left, top, left + cw, top + ch)).resize((width, height), Image.BILINEAR)

        if has_film:
            arr = np.array(cropped).astype(np.float32)
            sepia = _apply_sepia(np.array(cropped)).astype(np.float32)
            arr = arr * 0.45 + sepia * 0.55
            arr = _apply_vignette(arr.astype(np.uint8), vignette_mask).astype(np.float32)
            grain = grain_pool[int(t * 4) % len(grain_pool)]
            arr = np.clip(arr + grain, 0, 255)
            arr = arr * (1 - bars_alpha) + bars_rgb * bars_alpha
        else:
            arr = np.array(cropped).astype(np.float32)

        return np.clip(arr, 0, 255).astype(np.uint8)

    return VideoClip(make_frame, duration=duration)


def generate_voice_from_script(script_text, voice_choice, progress=gr.Progress()):
    if not script_text or not script_text.strip():
        raise gr.Error("Please enter script text first.")
    text = script_text.strip()
    if len(text) > MAX_SCRIPT_CHARS:
        text = text[:MAX_SCRIPT_CHARS]
        gr.Warning(f"Script trimmed to first {MAX_SCRIPT_CHARS:,} characters.")
    voice_name = _shortname_from_label(voice_choice)
    progress(0.3, desc="Generating AI voice...")
    path = os.path.join(OUT_DIR, f"voice_{uuid.uuid4().hex[:8]}.mp3")

    async def _speak():
        await edge_tts.Communicate(text, voice_name).save(path)

    asyncio.run(_speak())
    progress(1.0, desc="Voice ready")
    return path


def _resolve_per_image_plan(image_paths, table_rows, default_effects, total_duration):
    row_by_name = {}
    if table_rows:
        for row in table_rows:
            if not row or len(row) < 1:
                continue
            name = str(row[0])
            try:
                dur = float(row[1]) if len(row) > 1 and row[1] not in (None, "") else 1.0
            except (ValueError, TypeError):
                dur = 1.0
            eff_text = str(row[2]).strip() if len(row) > 2 and row[2] else ""
            row_by_name[os.path.basename(name)] = (max(0.1, dur), eff_text)

    weights = []
    per_image_effects = []
    for p in image_paths:
        base = os.path.basename(p)
        if base in row_by_name:
            dur, eff_text = row_by_name[base]
            weights.append(dur)
            if eff_text:
                names = [e.strip() for e in eff_text.split(",") if e.strip()]
                keys = [EFFECT_OPTIONS[n] for n in names if n in EFFECT_OPTIONS]
                per_image_effects.append(keys if keys else default_effects)
            else:
                per_image_effects.append(default_effects)
        else:
            weights.append(1.0)
            per_image_effects.append(default_effects)

    total_weight = sum(weights) if weights else 1.0
    durations = [(w / total_weight) * total_duration for w in weights]
    return list(zip(image_paths, durations, per_image_effects))


def render_video(audio_file, script_text, uploaded_images, nature_scene, selected_effects,
                 caption_style, render_speed, aspect_ratio, google_font, caption_color,
                 show_border, text_case, progress=gr.Progress()):
    if not audio_file:
        raise gr.Error("Please provide an audio file or generate a voice first.")

    progress(0.1, desc="Loading Audio...")
    audio_clip = AudioFileClip(audio_file)
    total_duration = audio_clip.duration

    if total_duration > MAX_AUDIO_SECONDS:
        raise gr.Error(f"Audio is too long ({total_duration:.1f}s). Max limit is {MAX_AUDIO_SECONDS}s.")

    fps, width, height = _resolve_dimensions(render_speed, aspect_ratio)

    image_paths = [img.name for img in uploaded_images] if uploaded_images else []
    
    if not image_paths and nature_scene in NATURE_SCENES:
        progress(0.2, desc="Generating Nature Background...")
        gen_fn = NATURE_SCENES[nature_scene]
        pil_img = gen_fn(width, height)
        temp_img_path = os.path.join(OUT_DIR, f"nature_{uuid.uuid4().hex[:6]}.png")
        pil_img.save(temp_img_path)
        image_paths = [temp_img_path]

    if not image_paths:
        raise gr.Error("No images uploaded and no nature background selected.")

    mapped_effects = [EFFECT_OPTIONS[e] for e in (selected_effects or []) if e in EFFECT_OPTIONS]
    plan = _resolve_per_image_plan(image_paths, None, mapped_effects, total_duration)

    progress(0.3, desc="Processing Video Clips...")
    clips = []
    for idx, (path, dur, effs) in enumerate(plan):
        clip = _make_image_clip(path, width, height, dur, effs, seed=idx)
        clips.append(clip)

    video_base = concatenate_videoclips(clips, method="compose")

    if script_text and script_text.strip() and caption_style != "None":
        progress(0.5, desc="Rendering Subtitles...")
        font_family = GOOGLE_FONTS.get(google_font)
        font_path = _download_google_font(font_family) if font_family else None
        
        words = script_text.strip().split()
        chunk_size = max(1, len(words) // len(plan))
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

        caption_clips = []
        cur_t = 0.0
        for i, chunk in enumerate(chunks[:len(plan)]):
            dur = plan[i][1]
            frame_arr = _make_caption_frame(
                chunk, width, height, style=CAPTION_STYLE_OPTIONS.get(caption_style, "classic"),
                font_path=font_path, custom_color=caption_color, border_on=show_border, text_case=text_case
            )
            cap_clip = ImageClip(frame_arr).set_start(cur_t).set_duration(dur)
            caption_clips.append(cap_clip)
            cur_t += dur

        final_video = CompositeVideoClip([video_base] + caption_clips)
    else:
        final_video = video_base

    final_video = final_video.set_audio(audio_clip)

    out_path = os.path.join(OUT_DIR, f"generated_video_{uuid.uuid4().hex[:8]}.mp4")
    progress(0.7, desc="Exporting Video MP4...")

    logger = GradioRenderLogger(progress, start=0.7, end=0.98)
    final_video.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        logger=logger,
    )

    progress(1.0, desc="Finished!")
    return out_path


def build_app():
    all_locales = _all_locales()
    friendly_languages = ["All Languages"] + sorted(list(set(_locale_label(loc) for loc in all_locales)))

    with gr.Blocks(title="Auto Video Generator") as app:
        gr.Markdown("# 🎬 Auto Video Generator")
        gr.Markdown("Turn voice scripts and images into stylized videos automatically.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Audio Source")
                audio_input = gr.Audio(label="Upload Audio File", type="filepath")

                with gr.Accordion("OR Generate AI Voice", open=False):
                    script_input = gr.Textbox(label="Script Text", lines=4, placeholder="Type your script here...")
                    lang_dropdown = gr.Dropdown(choices=friendly_languages, value="English (US)", label="Language")
                    gender_dropdown = gr.Dropdown(choices=["All", "Male", "Female"], value="All", label="Gender")
                    voice_dropdown = gr.Dropdown(
                        choices=_filtered_voice_choices("English (US)", "All", False),
                        value=_filtered_voice_choices("English (US)", "All", False)[0],
                        label="Select Voice",
                    )
                    gen_voice_btn = gr.Button("Generate AI Voice")

                gr.Markdown("### 2. Images / Background")
                image_input = gr.File(label="Upload Images", file_count="multiple", file_types=["image"])
                nature_dropdown = gr.Dropdown(choices=["None"] + list(NATURE_SCENES.keys()), value="None", label="Procedural Background (Fallback)")

                gr.Markdown("### 3. Visual Effects & Format")
                effects_checkbox = gr.CheckboxGroup(choices=list(EFFECT_OPTIONS.keys()), value=["Zoom (Ken Burns)"], label="Camera Effects")
                aspect_ratio = gr.Dropdown(choices=list(ASPECT_RATIO_OPTIONS.keys()), value="16:9 (YouTube Landscape)", label="Aspect Ratio")
                render_speed = gr.Dropdown(choices=list(RENDER_SPEED_OPTIONS.keys()), value="Balanced (recommended)", label="Render Quality/Speed")

                gr.Markdown("### 4. Captions & Styling")
                caption_style = gr.Dropdown(choices=["None"] + list(CAPTION_STYLE_OPTIONS.keys()), value="Classic Bottom", label="Caption Style")
                google_font = gr.Dropdown(choices=list(GOOGLE_FONTS.keys()), value="Poppins", label="Google Font")
                caption_color = gr.ColorPicker(value="#FFFFFF", label="Text Color")
                show_border = gr.Checkbox(value=True, label="Text Outline / Shadow")
                text_case = gr.Radio(choices=["Normal", "UPPERCASE", "lowercase"], value="Normal", label="Text Case")

                generate_btn = gr.Button("🚀 Render Video", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Preview & Download")
                video_output = gr.Video(label="Generated Video")

        def update_voices(lang, gender):
            choices = _filtered_voice_choices(lang, gender, False)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        lang_dropdown.change(update_voices, inputs=[lang_dropdown, gender_dropdown], outputs=[voice_dropdown])
        gender_dropdown.change(update_voices, inputs=[lang_dropdown, gender_dropdown], outputs=[voice_dropdown])

        gen_voice_btn.click(
            generate_voice_from_script,
            inputs=[script_input, voice_dropdown],
            outputs=[audio_input],
        )

        generate_btn.click(
            render_video,
            inputs=[
                audio_input, script_input, image_input, nature_dropdown,
                effects_checkbox, caption_style, render_speed, aspect_ratio,
                google_font, caption_color, show_border, text_case
            ],
            outputs=[video_output],
        )

    return app


if __name__ == "__main__":
    demo = build_app()
    # Required Railway configuration: listen on 0.0.0.0 and dynamically bind PORT
    port = int(os.environ.get("PORT", 7860))
    demo.queue().launch(server_name="0.0.0.0", server_port=port)
