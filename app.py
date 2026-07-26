"""
Auto Video Generator - Gradio app for Hugging Face Spaces.

Flow:
  1. Voice: upload your own audio, OR type a script and generate an AI voice
     (edge-tts, male/female/accent choice).
  2. Images: upload one or more images.
  3. Effect: pick from 5 looks - None (static), Zoom (Ken Burns), Pan Drift,
     Camera Shake, or Film Frame (vintage border + grain + vignette).
  4. Captions: optionally burn the script text on screen in one of 4 styles,
     timed across the video.
  5. Generate: renders an MP4 you can preview and download.
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
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.audio.fx.all import audio_loop, volumex

OUT_DIR = os.path.join(tempfile.gettempdir(), "auto_video_gen")
os.makedirs(OUT_DIR, exist_ok=True)

MAX_DIMENSION = 854
MAX_SCRIPT_CHARS = 500_000
WARN_AUDIO_SECONDS = 120
MAX_AUDIO_SECONDS = 3600

FAVORITES_PATH = os.path.join(OUT_DIR, "favorite_voices.json")

# Friendly display names for common locale codes; anything missing just falls
# back to showing the raw locale code (e.g. "xx-YY").
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
    """Returns a list of dicts: ShortName, Gender, Locale. Cached to disk so we
    only hit the edge-tts voice list endpoint once per environment."""
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
    locales = sorted(set(v["Locale"] for v in voices))
    return locales


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


def _save_favorites(fav_list):
    try:
        with open(FAVORITES_PATH, "w", encoding="utf-8") as f:
            json.dump(fav_list, f)
    except Exception:
        pass


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
    return out


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

SLOW_EFFECTS = {"kenburns", "pan", "shake", "filmframe", "rain"}

CAPTION_STYLE_OPTIONS = {
    "Classic Bottom": "classic",
    "Bold Outline": "outline",
    "Boxed": "boxed",
    "Minimal Top": "minimal",
}

# (fps, long_side_pixels) - lower values render faster on free CPU
RENDER_SPEED_OPTIONS = {
    "Fast (quick render, lower quality)": (12, 640),
    "Balanced (recommended)": (16, 854),
    "High Quality (slow)": (24, 1280),
}

# (width_ratio, height_ratio)
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


# ---------------------------------------------------------------------------
# Built-in nature backgrounds — drawn entirely with PIL, no external image
# files needed, so "Nature" mode works without uploading any pictures.
# ---------------------------------------------------------------------------

def _vertical_gradient(w, h, top_rgb, bottom_rgb):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        r = int(top_rgb[0] + (bottom_rgb[0] - top_rgb[0]) * t)
        g = int(top_rgb[1] + (bottom_rgb[1] - top_rgb[1]) * t)
        b = int(top_rgb[2] + (bottom_rgb[2] - top_rgb[2]) * t)
        for x in range(0, w, 4):  # draw in 4px vertical stripes: much faster, no visible banding
            px[x, y] = (r, g, b)
    img = img.resize((w, h))
    return img


def _generate_sky_scene(w, h):
    img = _vertical_gradient(w, h, (94, 170, 235), (214, 238, 252))
    draw = ImageDraw.Draw(img, "RGBA")
    sun_x = random.uniform(0.62, 0.86)
    sun_r = int(h * random.uniform(0.07, 0.11))
    draw.ellipse([w * sun_x - sun_r, h * 0.16 - sun_r, w * sun_x + sun_r, h * 0.16 + sun_r], fill=(255, 250, 214, 235))
    n_clouds = random.randint(3, 6)
    for _ in range(n_clouds):
        cx, cy, s = random.uniform(0.05, 0.95), random.uniform(0.1, 0.45), random.uniform(0.5, 1.1)
        cw, ch = int(w * 0.16 * s), int(h * 0.05 * s)
        x, y = int(w * cx), int(h * cy)
        for dx, dy, sc in [(-cw * 0.5, 0, 0.7), (0, -ch * 0.3, 1.0), (cw * 0.5, 0, 0.75)]:
            draw.ellipse([x + dx - cw * sc / 2, y + dy - ch * sc / 2, x + dx + cw * sc / 2, y + dy + ch * sc / 2], fill=(255, 255, 255, 210))
    return img


def _draw_mountain_layers(draw, w, h, base_top, base_bottom, colors, jitter=(0.06, 0.22)):
    n = len(colors)
    for i, color in enumerate(colors):
        base_frac = base_top + (base_bottom - base_top) * (i + 1) / n
        base_y = h * base_frac
        pts = [(0, h)]
        x, step = 0, w / random.randint(6, 9)
        while x <= w + step:
            peak = base_y - random.uniform(*jitter) * h
            pts.append((x, peak))
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
    river = Image.new("RGB", (w, h - 2 * bank_h), (72, 138, 178))
    rpx = river.load()
    rh = river.height
    phase = random.uniform(0, 6.28)
    for y in range(rh):
        shade = int(20 * math.sin(y * 0.15 + phase))
        for x in range(0, w, 3):
            rpx[x, y] = (72 + shade, 138 + shade, 178 + shade)
    river = river.resize((w, rh))
    img.paste(river, (0, bank_h))
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(14):
        y = bank_h + int((i / 14) * (h - 2 * bank_h)) + 6
        draw.line([(0, y), (w, y + 8)], fill=(255, 255, 255, 60), width=2)
    return img


def _generate_sunset_scene(w, h):
    img = _vertical_gradient(w, h, (70, 40, 90), (255, 150, 90))
    draw = ImageDraw.Draw(img, "RGBA")
    sun_r = int(h * 0.13)
    sun_y = h * 0.62
    draw.ellipse([w * 0.5 - sun_r, sun_y - sun_r, w * 0.5 + sun_r, sun_y + sun_r], fill=(255, 214, 130, 255))
    _draw_mountain_layers(draw, w, h, 0.72, 1.0, [(60, 30, 55), (35, 16, 40)], jitter=(0.03, 0.1))
    return img


def _generate_forest_scene(w, h):
    img = _vertical_gradient(w, h, (150, 210, 225), (225, 245, 235))
    draw = ImageDraw.Draw(img, "RGBA")
    ground_h = int(h * 0.12)
    draw.rectangle([0, h - ground_h, w, h], fill=(90, 130, 70))
    rows = [(0.55, (100, 140, 95), 0.09), (0.7, (65, 105, 70), 0.11), (0.86, (35, 70, 48), 0.13)]
    for base_frac, color, tw in rows:
        base_y = h * base_frac
        tree_w = w * tw
        x = -tree_w * random.uniform(0, 1)
        while x < w:
            tx = x + tree_w * random.uniform(0.7, 1.3)
            th = h * random.uniform(0.16, 0.26)
            draw.polygon([(tx, base_y - th), (tx - tree_w / 2, base_y), (tx + tree_w / 2, base_y)], fill=color)
            x = tx + tree_w * 0.5
    return img


def _generate_ocean_scene(w, h):
    img = _vertical_gradient(w, h, (120, 190, 235), (200, 235, 245))
    draw = ImageDraw.Draw(img, "RGBA")
    horizon = int(h * 0.42)
    sea = _vertical_gradient(w, h - horizon, (40, 120, 165), (100, 175, 205))
    img.paste(sea, (0, horizon))
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(10):
        y = horizon + int((i / 10) * (h - horizon) * 0.85) + 10
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 70), width=1)
    sand_h = int(h * 0.08)
    draw.rectangle([0, h - sand_h, w, h], fill=(230, 210, 165))
    return img


def _generate_desert_scene(w, h):
    img = _vertical_gradient(w, h, (255, 210, 140), (255, 240, 200))
    draw = ImageDraw.Draw(img, "RGBA")
    sun_r = int(h * 0.09)
    draw.ellipse([w * 0.72 - sun_r, h * 0.22 - sun_r, w * 0.72 + sun_r, h * 0.22 + sun_r], fill=(255, 245, 210, 255))
    _draw_mountain_layers(draw, w, h, 0.66, 1.0, [(225, 170, 110), (205, 145, 90), (180, 118, 70)], jitter=(0.02, 0.08))
    return img


def _generate_snowy_scene(w, h):
    img = _vertical_gradient(w, h, (185, 210, 235), (235, 245, 250))
    draw = ImageDraw.Draw(img, "RGBA")
    _draw_mountain_layers(draw, w, h, 0.5, 1.0, [(150, 160, 178), (110, 120, 145), (75, 85, 112)])
    for _ in range(random.randint(3, 5)):
        cx = random.uniform(0.1, 0.9) * w
        cy = h * random.uniform(0.42, 0.6)
        cap_w = w * random.uniform(0.07, 0.12)
        draw.polygon([(cx, cy), (cx - cap_w / 2, cy + cap_w * 0.55), (cx + cap_w / 2, cy + cap_w * 0.55)], fill=(255, 255, 255, 235))
    return img


def _generate_night_scene(w, h):
    img = _vertical_gradient(w, h, (10, 12, 40), (45, 40, 85))
    draw = ImageDraw.Draw(img, "RGBA")
    moon_r = int(h * 0.08)
    draw.ellipse([w * 0.78 - moon_r, h * 0.18 - moon_r, w * 0.78 + moon_r, h * 0.18 + moon_r], fill=(240, 240, 225, 255))
    for _ in range(int(w * h / 4000)):
        x, y = random.uniform(0, w), random.uniform(0, h * 0.75)
        r = random.uniform(0.5, 1.6)
        a = random.randint(120, 255)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, a))
    _draw_mountain_layers(draw, w, h, 0.78, 1.0, [(20, 18, 35), (10, 9, 20)], jitter=(0.02, 0.07))
    return img


def _generate_lake_scene(w, h):
    top = _generate_mountain_scene(w, h // 2)
    reflection = top.transpose(Image.FLIP_TOP_BOTTOM).convert("RGBA")
    tint = Image.new("RGBA", reflection.size, (60, 110, 160, 70))
    reflection = Image.alpha_composite(reflection, tint)
    img = Image.new("RGB", (w, h))
    img.paste(top, (0, 0))
    img.paste(reflection.convert("RGB"), (0, h // 2))
    draw = ImageDraw.Draw(img, "RGBA")
    for i in range(10):
        y = h // 2 + int((i / 10) * (h - h // 2))
        draw.line([(0, y), (w, y)], fill=(255, 255, 255, 40), width=1)
    return img


NATURE_SCENES = {
    "Sky": _generate_sky_scene,
    "Mountains": _generate_mountain_scene,
    "River": _generate_river_scene,
    "Sunset": _generate_sunset_scene,
    "Forest": _generate_forest_scene,
    "Ocean Beach": _generate_ocean_scene,
    "Desert": _generate_desert_scene,
    "Snowy Peaks": _generate_snowy_scene,
    "Night Sky": _generate_night_scene,
    "Lake Reflection": _generate_lake_scene,
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

# Curated Google Fonts for caption styling - one popular weight each.
GOOGLE_FONTS = {
    "Default (built-in)": None,
    "Poppins": "Poppins",
    "Roboto": "Roboto",
    "Montserrat": "Montserrat",
    "Oswald": "Oswald",
    "Bebas Neue": "Bebas Neue",
    "Anton": "Anton",
    "Playfair Display": "Playfair Display",
    "Lobster": "Lobster",
    "Pacifico": "Pacifico",
    "Dancing Script": "Dancing Script",
    "Raleway": "Raleway",
    "Nunito": "Nunito",
    "Open Sans": "Open Sans",
    "Merriweather": "Merriweather",
    "Bangers": "Bangers",
    "Righteous": "Righteous",
    "Permanent Marker": "Permanent Marker",
    "Caveat": "Caveat",
    "Archivo Black": "Archivo Black",
    "Inter": "Inter",
}


def _download_google_font(font_family, weight=700):
    """Downloads a Google Font ttf and caches it locally. Returns a filepath,
    or None if the font couldn't be fetched (falls back to built-in font)."""
    if not font_family:
        return None
    cache_path = os.path.join(FONT_CACHE_DIR, f"{font_family.replace(' ', '_')}_{weight}.ttf")
    if os.path.exists(cache_path):
        return cache_path
    try:
        css_url = f"https://fonts.googleapis.com/css2?family={font_family.replace(' ', '+')}:wght@{weight}&display=swap"
        # An old User-Agent makes Google serve legacy .ttf instead of .woff2
        headers = {"User-Agent": "Mozilla/4.0 (compatible; MSIE 5.0; Windows 98)"}
        resp = requests.get(css_url, headers=headers, timeout=15)
        match = re.search(r"url\((https://fonts\.gstatic\.com/[^)]+\.ttf)\)", resp.text)
        if not match:
            return None
        ttf_bytes = requests.get(match.group(1), timeout=15).content
        with open(cache_path, "wb") as f:
            f.write(ttf_bytes)
        return cache_path
    except Exception:
        return None


def _make_font_preview(font_choice, hex_color, border_on, text_case):
    font_family = GOOGLE_FONTS.get(font_choice)
    font_path = _download_google_font(font_family) if font_family else None
    sample = "The quick brown fox"
    sample = sample.upper() if text_case == "UPPERCASE" else (sample.lower() if text_case == "lowercase" else sample)
    w, h = 640, 160
    frame = Image.new("RGBA", (w, h), (30, 30, 30, 255))
    draw = ImageDraw.Draw(frame)
    font = ImageFont.truetype(font_path, 48) if font_path else _load_font(48)
    fill = _hex_to_rgba(hex_color)
    bbox = draw.textbbox((0, 0), sample, font=font)
    tx = (w - (bbox[2] - bbox[0])) // 2
    ty = (h - (bbox[3] - bbox[1])) // 2
    if border_on:
        for dx in range(-3, 4, 3):
            for dy in range(-3, 4, 3):
                if dx or dy:
                    draw.text((tx + dx, ty + dy), sample, font=font, fill=(0, 0, 0, 255))
    draw.text((tx, ty), sample, font=font, fill=fill)
    return np.array(frame.convert("RGB"))


def _hex_to_rgba(hex_color, alpha=255):
    if not hex_color:
        return (255, 255, 255, alpha)
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255, alpha)
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (r, g, b, alpha)


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
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return img.crop((left, top, left + target_w, top + target_h))


def _apply_sepia(np_img):
    img = np_img.astype(np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    tr = 0.393 * r + 0.769 * g + 0.189 * b
    tg = 0.349 * r + 0.686 * g + 0.168 * b
    tb = 0.272 * r + 0.534 * g + 0.131 * b
    out = np.stack([tr, tg, tb], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def _make_vignette_mask(h, w):
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dist = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
    mask = np.clip(1.0 - (dist - 0.45) * 0.85, 0.35, 1.0)
    return mask


def _apply_vignette(np_img, mask):
    out = np_img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _add_grain(np_img, amount=10):
    noise = np.random.randint(-amount, amount + 1, np_img.shape[:2] + (1,))
    out = np_img.astype(np.int16) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_film_bars(pil_img):
    w, h = pil_img.size
    draw = ImageDraw.Draw(pil_img)
    bar_h = max(6, int(h * 0.055))
    draw.rectangle([(0, 0), (w, bar_h)], fill=(0, 0, 0))
    draw.rectangle([(0, h - bar_h), (w, h)], fill=(0, 0, 0))

    hole_w = max(4, int(bar_h * 0.55))
    hole_h = max(3, int(bar_h * 0.4))
    gap = max(hole_w * 2, int(hole_w * 1.8))
    y1 = int(bar_h * 0.3)
    y2 = h - bar_h + int(bar_h * 0.3)

    x = gap // 2
    while x < w:
        draw.rounded_rectangle([x - hole_w // 2, y1, x + hole_w // 2, y1 + hole_h], radius=3, fill=(235, 230, 222))
        draw.rounded_rectangle([x - hole_w // 2, y2, x + hole_w // 2, y2 + hole_h], radius=3, fill=(235, 230, 222))
        x += gap
    return pil_img


def _make_caption_frame(text, width, height, style="classic", font_path=None, custom_color=None,
                         border_on=None, text_case="Normal"):
    """style controls the background shape (classic bar / outline / boxed / minimal).
    font_path/custom_color/border_on/text_case (from the Subtitle Style Editor) override
    the style preset's default font, color, and border when provided."""
    if text_case == "UPPERCASE":
        text = text.upper()
    elif text_case == "lowercase":
        text = text.lower()

    max_block_h = int(height * (0.14 if style == "minimal" else 0.20))
    size_divisor = {"classic": 38, "outline": 28, "boxed": 34, "minimal": 44}.get(style, 38)
    font_size = max(13, min(width // size_divisor, height // 13))

    for _ in range(4):
        font = _load_font(font_size, font_path)
        wrapped = textwrap.fill(text, width=max(10, width // (font_size // 2)))
        lines = wrapped.split("\n")
        line_h = int(font_size * 1.3)
        block_h = line_h * len(lines) + 40
        if block_h <= max_block_h or font_size <= 14:
            break
        font_size = max(14, int(font_size * (max_block_h / block_h)))

    frame = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(frame)

    y = int(height * 0.06) if style == "minimal" else height - block_h - 30

    outline_fill = None
    outline_w = 0
    if style == "classic":
        draw.rectangle([(0, y - 20), (width, y + block_h)], fill=(0, 0, 0, 140))
        text_fill, outline_fill, outline_w = (255, 255, 255, 255), (0, 0, 0, 255), 2
    elif style == "outline":
        text_fill = (255, 210, 63, 255)
        outline_fill, outline_w = (0, 0, 0, 255), max(3, font_size // 10)
    elif style == "boxed":
        widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
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


def _target_size_from_image(path, max_dim=MAX_DIMENSION):
    with Image.open(path) as img:
        w, h = img.size
    scale = min(1.0, max_dim / max(w, h))
    w, h = int(w * scale), int(h * scale)
    w -= w % 2
    h -= h % 2
    return max(w, 2), max(h, 2)


def _split_into_chunks(script, n_chunks):
    words = script.split()
    if not words:
        return []
    n_chunks = max(1, n_chunks)
    per = max(1, len(words) // n_chunks)
    chunks = []
    for i in range(0, len(words), per):
        chunks.append(" ".join(words[i:i + per]))
    return chunks[:n_chunks] if len(chunks) > n_chunks else chunks


def _make_image_clip(path, width, height, duration, effects, seed=0):
    """effects: a list of effect keys (e.g. ["kenburns", "shake", "filmframe"]).
    All selected effects are combined together on the same clip."""
    img = Image.open(path)
    effects = [e for e in (effects or []) if e and e != "none"]

    if not effects:
        pil_img = _cover_resize(img, width, height)
        return ImageClip(np.array(pil_img)).set_duration(duration)

    has_kb = "kenburns" in effects
    has_pan = "pan" in effects
    has_shake = "shake" in effects
    has_film = "filmframe" in effects
    has_rain = "rain" in effects

    # Combined zoom margin: enough extra canvas for whichever motions are active.
    zoom_margin = 1.0
    if has_kb:
        zoom_margin += 0.16
    if has_pan:
        zoom_margin += 0.18
    if has_shake:
        zoom_margin += 0.08
    if has_film:
        zoom_margin += 0.08

    big_w, big_h = int(width * zoom_margin), int(height * zoom_margin)
    big_img = _cover_resize(img, big_w, big_h)

    pan_direction = 1 if seed % 2 == 0 else -1
    pan_room_x = (big_w - width) * 0.35
    shake_room_x, shake_room_y = (big_w - width) * 0.22, (big_h - height) * 0.22
    kb_zoom_amt = 0.16 if has_kb else 0.0
    film_zoom_amt = 0.06 if has_film else 0.0

    vignette_mask = _make_vignette_mask(height, width) if has_film else None
    bars_alpha = bars_rgb = grain_pool = None
    if has_film:
        bars_rgba = np.array(_draw_film_bars(Image.new("RGBA", (width, height), (0, 0, 0, 0))))
        bars_alpha = (bars_rgba[..., 3:4].astype(np.float32)) / 255.0
        bars_rgb = bars_rgba[..., :3].astype(np.float32)
        grain_pool = [np.random.randint(-5, 6, (height, width, 1)).astype(np.int16) for _ in range(8)]

    rain_tile = rain_tile_h = None
    if has_rain:
        rain_tile_h = height * 2
        tile_img = Image.new("RGBA", (width, rain_tile_h), (0, 0, 0, 0))
        rdraw = ImageDraw.Draw(tile_img)
        import random as _r
        rrng = _r.Random(seed * 101 + 3)
        drop_len = max(8, height // 22)
        for _ in range(int(width * rain_tile_h / 2200)):
            x = rrng.randint(0, width)
            y = rrng.randint(0, rain_tile_h)
            alpha = rrng.randint(60, 130)
            rdraw.line([(x, y), (x - drop_len * 0.35, y + drop_len)], fill=(220, 235, 245, alpha), width=1)
        rain_tile = np.array(tile_img)
        rain_alpha_full = (rain_tile[..., 3:4].astype(np.float32)) / 255.0
        rain_rgb_full = rain_tile[..., :3].astype(np.float32)

    def make_frame(t):
        progress = min(1.0, t / duration) if duration > 0 else 0.0

        # zoom: kenburns + film-frame's mild zoom stack additively, shrinking crop over time
        zoom_amt = kb_zoom_amt + film_zoom_amt
        scale = (1 + zoom_amt) - zoom_amt * progress
        cw, ch = width * scale, height * scale

        dx = dy = 0.0
        if has_pan:
            dx += pan_direction * (progress - 0.5) * pan_room_x * 2
        if has_shake:
            dx += math.sin(t * 1.6 + seed * 3) * shake_room_x * 0.7 + math.sin(t * 3.1 + seed) * shake_room_x * 0.25
            dy += math.cos(t * 1.9 + seed * 2) * shake_room_y * 0.7 + math.cos(t * 2.6 + seed) * shake_room_y * 0.25
        if has_film:
            dx += math.sin(t * 1.3 + seed * 4) * shake_room_x * 0.25
            dy += math.cos(t * 1.6 + seed * 2) * shake_room_y * 0.25

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

        if has_rain:
            offset = int(t * 260) % height
            alpha_win = rain_alpha_full[offset:offset + height]
            rgb_win = rain_rgb_full[offset:offset + height]
            arr = arr * (1 - alpha_win) + rgb_win * alpha_win

        return np.clip(arr, 0, 255).astype(np.uint8)

    return VideoClip(make_frame, duration=duration)


def generate_voice_from_script(script_text, voice_choice, progress=gr.Progress()):
    if not script_text or not script_text.strip():
        raise gr.Error("Pehle script text likhein.")
    text = script_text.strip()
    if len(text) > MAX_SCRIPT_CHARS:
        text = text[:MAX_SCRIPT_CHARS]
        gr.Warning(f"Script {len(script_text.strip())} characters ka tha, pehle {MAX_SCRIPT_CHARS:,} characters hi use kiye gaye.")
    voice_name = _shortname_from_label(voice_choice)
    progress(0.3, desc="Generating AI voice...")
    path = os.path.join(OUT_DIR, f"voice_{uuid.uuid4().hex[:8]}.mp3")

    async def _speak():
        await edge_tts.Communicate(text, voice_name).save(path)

    asyncio.run(_speak())
    progress(1.0, desc="Voice ready")
    return path


def _resolve_per_image_plan(image_paths, table_rows, default_effects, total_duration):
    """table_rows: list of [filename, duration_seconds, effects_text] from the images
    Dataframe (may be shorter/out of order vs image_paths, or empty). Returns a list of
    (path, duration, effects_list) whose durations are scaled so they sum exactly to
    total_duration, preserving the *relative* pacing the user set per image."""
    row_by_name = {}
    if table_rows:
        for row in table_rows:
            if not row:
                continue
            name = str(row[0]) if len(row) > 0 else None
            if not name:
                continue
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

    total_weight = sum(weights) or 1.0
    durations = [total_duration * (w / total_weight) for w in weights]
    return list(zip(image_paths, durations, per_image_effects))


def generate_video(script_text, voice_audio, images, images_table, effect_choices, add_captions, caption_style_choice,
                    subtitle_border, subtitle_case, subtitle_color, subtitle_font,
                    speed_choice, aspect_choice, bg_source, nature_choices, randomize_nature, progress=gr.Progress()):
    use_nature = bg_source == "Nature scenes (image upload zaroori nahi)"
    if not use_nature and not images:
        raise gr.Error("Kam se kam ek image upload karein, ya 'Nature scenes' background source chunein.")
    if use_nature and not randomize_nature and not nature_choices:
        raise gr.Error("Kam se kam ek Nature scene select karein, ya 'Random scenes' checkbox on karein.")
    if voice_audio is None:
        raise gr.Error("Pehle voice ready karein - 'Generate AI Voice from Script' button dabayein, ya khud ka audio upload karein.")

    run_id = uuid.uuid4().hex[:8]
    effects = [EFFECT_OPTIONS[e] for e in (effect_choices or []) if e in EFFECT_OPTIONS]
    caption_style = CAPTION_STYLE_OPTIONS.get(caption_style_choice, "classic")
    fps, width, height = _resolve_dimensions(speed_choice, aspect_choice)

    progress(0.05, desc="Preparing audio...")
    audio_clip = AudioFileClip(voice_audio)
    if audio_clip.duration > MAX_AUDIO_SECONDS:
        audio_clip = audio_clip.subclip(0, MAX_AUDIO_SECONDS)
        gr.Warning(f"Audio 1 hour se lamba tha, pehle {MAX_AUDIO_SECONDS // 60} minutes hi use kiye gaye.")
    elif audio_clip.duration > WARN_AUDIO_SECONDS:
        minutes = round(audio_clip.duration / 60, 1)
        gr.Warning(f"Audio ~{minutes} min ka hai - poora video isi length ka banega. Free CPU par ye render hone me kaafi der (kayi minute se ghante tak) lag sakta hai.")
    total_duration = audio_clip.duration

    progress(0.25, desc="Preparing images...")
    if use_nature:
        if randomize_nature:
            all_scene_names = list(NATURE_SCENES.keys())
            pick_n = random.randint(3, min(6, len(all_scene_names)))
            scene_names = random.sample(all_scene_names, pick_n)
        else:
            scene_names = nature_choices
        image_paths = []
        for i, scene_name in enumerate(scene_names):
            gen_fn = NATURE_SCENES.get(scene_name)
            if not gen_fn:
                continue
            scene_img = gen_fn(width, height)
            scene_path = os.path.join(OUT_DIR, f"nature_{run_id}_{i}.png")
            scene_img.save(scene_path)
            image_paths.append(scene_path)
    else:
        image_paths = [im if isinstance(im, str) else im.name for im in images]

    image_plan = _resolve_per_image_plan(image_paths, images_table, effects, total_duration)

    if any(e in SLOW_EFFECTS for e in effects) and total_duration > WARN_AUDIO_SECONDS:
        gr.Warning("Lambe video ke liye motion/vintage effects 'None (static)' se zyada slow render karte hain - agar render bahut der le raha hai to sirf 'None (static)' chunein.")

    clips = [_make_image_clip(p, width, height, dur, per_effects, seed=i) for i, (p, dur, per_effects) in enumerate(image_plan)]

    video = concatenate_videoclips(clips, method="compose") if len(clips) > 1 else clips[0]
    video = video.set_audio(audio_clip)

    if add_captions and script_text and script_text.strip():
        progress(0.55, desc="Adding captions...")
        font_family = GOOGLE_FONTS.get(subtitle_font)
        font_path = _download_google_font(font_family) if font_family else None
        approx_n = max(1, int(total_duration // 3))
        chunks = _split_into_chunks(script_text.strip(), approx_n)
        if chunks:
            chunk_dur = total_duration / len(chunks)
            caption_clips = []
            for i, chunk in enumerate(chunks):
                frame = _make_caption_frame(
                    chunk, width, height, style=caption_style,
                    font_path=font_path, custom_color=subtitle_color,
                    border_on=subtitle_border, text_case=subtitle_case,
                )
                cap_clip = ImageClip(frame, transparent=True).set_start(i * chunk_dur).set_duration(chunk_dur)
                caption_clips.append(cap_clip)
            video = CompositeVideoClip([video] + caption_clips, size=(width, height))

    progress(0.7, desc="Rendering video (this can take a minute)...")
    out_path = os.path.join(OUT_DIR, f"output_{run_id}.mp4")
    video.write_videofile(
        out_path,
        fps=fps,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=2,
        ffmpeg_params=["-crf", "30"],
        logger=GradioRenderLogger(progress),
    )
    progress(1.0, desc="Done!")

    return out_path, out_path


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,600;0,700;1,600;1,700;1,800&display=swap');

body, .gradio-container, label, .block-title, span.svelte-1gfkn6j,
textarea, input, select, button, .prose, .prose p {
  font-family: 'Poppins', sans-serif !important;
  font-style: italic !important;
}

body, .gradio-container { font-size: 15px !important; }
h1 {
  font-size: 25px !important;
  font-weight: 800 !important;
  letter-spacing: 0.02em !important;
}
label, .block-title, span.svelte-1gfkn6j { font-size: 14px !important; font-weight: 600 !important; }
textarea, input, select { font-size: 15px !important; }
button { font-size: 14.5px !important; font-weight: 700 !important; padding: 11px !important; }
.gr-button-primary { font-size: 16px !important; font-weight: 800 !important; }
.prose, .prose p { font-size: 13.5px !important; line-height: 1.5 !important; }
"""

WHATSAPP_NUMBER = "0310-6206972"

with gr.Blocks(title="REHAN PRO", css=CUSTOM_CSS) as demo:
    gr.Markdown("# 🎬 REHAN PRO\nVoice ready karein (AI se generate karein ya khud upload karein) + images upload karein → effect aur caption style choose karein → video generate karke download karein.")

    with gr.Row():
        with gr.Column(scale=1):
            script_text = gr.Textbox(label="Script / Caption Text", placeholder="Yahan apna script likhein - ye AI voice aur captions dono ke liye use hoga.", lines=6)
            gr.Markdown("_Script max 500,000 characters tak, aur audio/video max **1 hour** tak use hoga. Itna lamba video free CPU par render hone me kaafi der (minutes se ghante tak) lag sakta hai - lambe video ke liye 'None (static)' effect fast rahega._")

            gr.Markdown("### 🔊 Voice")
            with gr.Row():
                gender_choice = gr.Radio(["All", "Female", "Male"], value="All", label="Gender")
                favorites_only = gr.Checkbox(label="⭐ Favorites only", value=False)
            language_choice = gr.Dropdown(
                ["All Languages"] + sorted(set(_locale_label(loc) for loc in _all_locales())),
                value="All Languages", label="Language",
            )
            voice_choice = gr.Dropdown(
                _filtered_voice_choices("All Languages", "All", False),
                value=(_filtered_voice_choices("All Languages", "All", False) or [None])[0],
                label="Voice (all Edge-TTS voices)",
            )
            with gr.Row():
                fav_toggle_btn = gr.Button("⭐ Add/Remove Favorite")
                generate_voice_btn = gr.Button("🔊 Generate AI Voice from Script", variant="secondary")
            voice_audio = gr.Audio(label="Voice (AI se generate hui, ya yahan khud apna audio upload karein)", type="filepath")

            gr.Markdown("### 🖼️ Images")
            bg_source = gr.Radio(
                ["Apni images upload karein", "Nature scenes (image upload zaroori nahi)"],
                value="Apni images upload karein",
                label="Background Source",
            )
            images = gr.File(label="Upload image(s)", file_count="multiple", file_types=["image"])
            load_table_btn = gr.Button("📋 Load Images into Duration/Effects Table")
            images_table = gr.Dataframe(
                headers=["Image (filename)", "Duration weight (sec)", "Effects (comma-separated, optional)"],
                datatype=["str", "number", "str"],
                row_count=(0, "dynamic"), col_count=(3, "fixed"), interactive=True,
                label="Per-image duration & motion effects",
            )
            gr.Markdown("_Har image ki 'Duration weight' uski relative length set karti hai - final video hamesha voice/audio ki poori length ka banega, weights ke hisaab se stretch/compress ho jayenge. **Effects** column mein comma se effect names likhein (jaise `Zoom (Ken Burns), Camera Shake`) - khaali chhodne par neeche wale global Effect(s) use honge._")
            randomize_nature = gr.Checkbox(label="Har video mein random scenes chunein (alag-alag, surprise me)", value=True, visible=False)
            nature_choices = gr.CheckboxGroup(
                list(NATURE_SCENES.keys()), value=["Mountains", "Sunset"],
                label="Nature Scene(s) manually chunein (sirf tab jab Random OFF ho)",
                visible=False,
            )
            gr.Markdown("_10 built-in scenes: Sky, Mountains, River, Sunset, Forest, Ocean Beach, Desert, Snowy Peaks, Night Sky, Lake Reflection - seedhe code se banti hain (koi photo upload nahi hoti). **Random** on rahe to har video mein alag scenes aayenge. **Rain Overlay** effect kisi bhi scene ke saath combine ho sakta hai._")

            aspect_choice = gr.Radio(list(ASPECT_RATIO_OPTIONS.keys()), value="16:9 (YouTube Landscape)", label="Aspect Ratio")

            effect = gr.CheckboxGroup(list(EFFECT_OPTIONS.keys()), value=["None (static)"], label="Default Effect(s) - jab table mein Effects khaali ho to ye lagenge")
            gr.Markdown("_Video aapke chune hue Aspect Ratio mein banegi - image/scene automatically crop ho kar fit ho jayegi. **Camera Shake** halka aur slow jhatka deta hai, **Film Frame** vintage sepia + halka grain + vignette deta hai, **Rain Overlay** girti hui baarish ki lines add karta hai._")

            gr.Markdown("### 💬 Subtitles")
            add_captions = gr.Checkbox(label="Burn captions on video (on/off)", value=True)
            caption_style = gr.Dropdown(list(CAPTION_STYLE_OPTIONS.keys()), value="Classic Bottom", label="Caption Style (background shape)")
            with gr.Accordion("✏️ Edit Subtitle Style", open=False) as subtitle_editor:
                subtitle_border = gr.Checkbox(label="Border/outline on text", value=True)
                subtitle_case = gr.Radio(["Normal", "UPPERCASE", "lowercase"], value="Normal", label="Letter Case")
                subtitle_color = gr.ColorPicker(label="Text Color", value="#FFFFFF")
                subtitle_font = gr.Dropdown(list(GOOGLE_FONTS.keys()), value="Default (built-in)", label="Font (Google Fonts, downloaded + cached on first use)")
                preview_font_btn = gr.Button("👁️ Preview Font Style")
                font_preview_img = gr.Image(label="Style Preview", interactive=False)

            speed_choice = gr.Radio(list(RENDER_SPEED_OPTIONS.keys()), value="Fast (quick render, lower quality)", label="Render Speed")
            gr.Markdown("_Lambe audio (jaise 10+ minute) ke saath koi bhi zoom/shake/pan/film/rain effect free CPU par kaafi der lega - **Fast** speed ya **None (static)** effect render ko sabse zyada tez karte hain._")

            generate_btn = gr.Button("🚀 Generate Video", variant="primary")

        with gr.Column(scale=1):
            video_out = gr.Video(label="Preview")
            download_out = gr.File(label="Download video")
            gr.Markdown(f"---\n📱 **Need help or custom work? WhatsApp: {WHATSAPP_NUMBER}**")

    def _toggle_bg_inputs(choice):
        is_nature = choice == "Nature scenes (image upload zaroori nahi)"
        return gr.update(visible=not is_nature), gr.update(visible=is_nature), gr.update(visible=is_nature)

    bg_source.change(_toggle_bg_inputs, inputs=bg_source, outputs=[images, randomize_nature, nature_choices])

    def _refresh_voice_choices(language_val, gender_val, favs_only_val):
        choices = _filtered_voice_choices(language_val, gender_val, favs_only_val)
        return gr.update(choices=choices, value=(choices[0] if choices else None))

    language_choice.change(_refresh_voice_choices, inputs=[language_choice, gender_choice, favorites_only], outputs=voice_choice)
    gender_choice.change(_refresh_voice_choices, inputs=[language_choice, gender_choice, favorites_only], outputs=voice_choice)
    favorites_only.change(_refresh_voice_choices, inputs=[language_choice, gender_choice, favorites_only], outputs=voice_choice)

    def _toggle_favorite(voice_label):
        shortname = _shortname_from_label(voice_label)
        favs = _load_favorites()
        if shortname in favs:
            favs.remove(shortname)
            gr.Info(f"Favorites se hata diya: {shortname}")
        else:
            favs.append(shortname)
            gr.Info(f"Favorites mein add kiya: {shortname}")
        _save_favorites(favs)

    fav_toggle_btn.click(_toggle_favorite, inputs=voice_choice, outputs=None)

    def _populate_images_table(files):
        if not files:
            return []
        rows = []
        for f in files:
            name = os.path.basename(f if isinstance(f, str) else f.name)
            rows.append([name, 3.0, ""])
        return rows

    load_table_btn.click(_populate_images_table, inputs=images, outputs=images_table)

    add_captions.change(lambda on: gr.update(visible=on), inputs=add_captions, outputs=subtitle_editor)

    preview_font_btn.click(
        _make_font_preview,
        inputs=[subtitle_font, subtitle_color, subtitle_border, subtitle_case],
        outputs=font_preview_img,
    )

    generate_voice_btn.click(generate_voice_from_script, inputs=[script_text, voice_choice], outputs=voice_audio)

    generate_btn.click(
        generate_video,
        inputs=[
            script_text, voice_audio, images, images_table, effect, add_captions, caption_style,
            subtitle_border, subtitle_case, subtitle_color, subtitle_font,
            speed_choice, aspect_choice, bg_source, nature_choices, randomize_nature,
        ],
        outputs=[video_out, download_out],
    )

if __name__ == "__main__":
    demo.queue().launch()
