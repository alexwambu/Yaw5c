# main.py
import os
import uuid
import asyncio
import shutil
import tempfile
from typing import List, Optional, Dict
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from moviepy.editor import (
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
)
from gtts import gTTS
from pydub import AudioSegment, silence

# --- Config / globals ---
JOB_STORE: Dict[str, dict] = {}
OUTPUT_DIR = "outputs"
TMP_DIR = "tmp"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Map character name -> voice provider key (default uses gTTS)
# You can override this map at runtime or provide external TTS credentials via env vars
CHAR_VOICE_MAP = {
    "NARRATOR": "gtts_en",
    # add more named voices like "ALICE": "eleven_alice"
}

# --- FastAPI app ---
app = FastAPI(title="Movie Generator with Multi-Character TTS + Preview + Auto-Edit")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# --- Utility: parse script with character tags ---
def parse_script_with_characters(script: str):
    """
    Expect script lines like:
      NARRATOR: The world was quiet.
      ALICE: Hello there!
      BOB: Hi.
    Lines without ':' are appended to previous speaker.
    Returns list of scenes: [{character, text}]
    """
    scenes = []
    current = None
    for raw in script.splitlines():
        line = raw.strip()
        if not line:
            # treat blank line as scene separator
            current = None
            continue
        if ":" in line:
            name, text = line.split(":", 1)
            name = name.strip().upper()
            text = text.strip()
            current = {"character": name, "text": text}
            scenes.append(current)
        else:
            # continuation of previous
            if current:
                current["text"] += " " + line
            else:
                # assign to NARRATOR by default
                current = {"character": "NARRATOR", "text": line}
                scenes.append(current)
    return scenes

# --- TTS layer: default gTTS wrapper + extension point for external APIs ---
def synthesize_with_gtts(text: str, out_path: str, lang: str = "en"):
    tts = gTTS(text=text, lang=lang)
    tts.save(out_path)
    return out_path

async def synthesize_character_voice(character: str, text: str, out_path: str):
    """
    Choose TTS provider based on CHAR_VOICE_MAP[character].
    For now we default to gTTS; replace this function to call ElevenLabs/Azure/etc.
    """
    provider = CHAR_VOICE_MAP.get(character.upper(), "gtts_en")
    if provider.startswith("gtts"):
        synthesize_with_gtts(text, out_path)
        return out_path
    else:
        # placeholder for other providers (ElevenLabs, Azure) if provider string matches
        # e.g. "eleven_alice" -> call ElevenLabs API with voice id "alice"
        # Implementations should save to out_path and return it.
        synthesize_with_gtts(text, out_path)
        return out_path

# --- Audio helpers using pydub ---
def concat_audios(audio_paths: List[str], out_path: str):
    if not audio_paths:
        # create short silent mp3
        silent = AudioSegment.silent(duration=1000)
        silent.export(out_path, format="mp3")
        return out_path

    combined = AudioSegment.empty()
    for p in audio_paths:
        seg = AudioSegment.from_file(p)
        combined += seg
    combined.export(out_path, format="mp3")
    return out_path

def detect_silence_ranges(audio_path: str, min_silence_len=400, silence_thresh=-40):
    seg = AudioSegment.from_file(audio_path)
    ranges = silence.detect_silence(seg, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    # ranges are [start_ms, end_ms] pairs
    return ranges

# --- Video helpers (ffmpeg via moviepy) ---
def image_to_video(image_path: str, out_video: str, duration: int = 5, resolution="1920x1080", fps=24):
    w, h = map(int, resolution.split("x"))
    clip = ImageClip(image_path).set_duration(duration).resize(height=h)
    # center & pad to resolution
    clip = clip.set_position(("center", "center"))
    # write using moviepy (ffmpeg)
    clip.write_videofile(out_video, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
    clip.close()
    return out_video

def clip_trim_or_loop(input_clip: str, out_clip: str, duration: int, resolution="1920x1080", fps=24):
    # use ffmpeg via moviepy for reliability
    v = VideoFileClip(input_clip)
    if v.duration >= duration:
        sub = v.subclip(0, duration)
        sub.write_videofile(out_clip, codec="libx264", fps=fps, audio_codec="aac", verbose=False, logger=None)
        sub.close()
    else:
        # loop
        times = int(duration // max(0.1, v.duration)) + 1
        clips = [v] * times
        final = concatenate_videoclips(clips).subclip(0, duration)
        final.write_videofile(out_clip, codec="libx264", fps=fps, audio_codec="aac", verbose=False, logger=None)
        final.close()
    v.close()
    return out_clip

def mux_video_with_audio(video_in: str, audio_in: str, out_path: str):
    video = VideoFileClip(video_in)
    audio = AudioFileClip(audio_in)
    video = video.set_audio(audio)
    video.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    video.close()
    audio.close()
    return out_path

def concat_videos(video_list: List[str], out_path: str):
    if not video_list:
        raise ValueError("No videos to concat")
    clips = [VideoFileClip(v) for v in video_list]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(out_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
    for c in clips:
        try:
            c.close()
        except:
            pass
    final.close()
    return out_path

# --- Preview helper: create a low-res sample of first N seconds ---
def generate_preview(full_video_path: str, preview_path: str, preview_seconds: int = 15, res="640x360"):
    v = VideoFileClip(full_video_path)
    dur = min(preview_seconds, v.duration)
    sub = v.subclip(0, dur)
    w, h = map(int, res.split("x"))
    sub = sub.resize(newsize=(w, h))
    sub.write_videofile(preview_path, codec="libx264", fps=24, audio_codec="aac", verbose=False, logger=None)
    sub.close()
    v.close()
    return preview_path

# --- Job worker: orchestrates scenes, voices, audio stitching, video assembly ---
async def worker_render_movie(job_id: str, script: str, images: List[str], clips: List[str], title: str, resolution="1920x1080", fps=24):
    """
    Steps & progress updates:
      0-5% validate
      5-15% parse scenes
      15-40% synthesize TTS per scene/character
      40-75% render scene videos & mux audio
      75-90% concat & postprocess
      90-99% preview & finalize
    """
    JOB_STORE[job_id]["progress"] = 5
    try:
        scenes = parse_script_with_characters(script)
        JOB_STORE[job_id]["progress"] = 10

        # group audio per scene (character voices)
        scene_audio_files = []
        JOB_STORE[job_id]["progress"] = 15

        for i, sc in enumerate(scenes):
            char = sc.get("character", "NARRATOR")
            text = sc.get("text", "")
            aud_path = os.path.join(TMP_DIR, f"{job_id}_scene_{i}_audio.mp3")
            await synthesize_character_voice(char, text, aud_path)
            scene_audio_files.append(aud_path)
            # update progress roughly
            JOB_STORE[job_id]["progress"] = 15 + int(20 * (i+1)/max(1,len(scenes)))

        # Optionally run simple "ML auto-edit" step: detect silence in audio and mark cuts
        JOB_STORE[job_id]["progress"] = 40
        # create scene videos: prefer user-supplied images/clips by round-robin
        scene_video_files = []
        for i, sc in enumerate(scenes):
            # get asset if available
            asset = None
            if images:
                asset = images[i % len(images)]
            elif clips:
                asset = clips[i % len(clips)]

            tmp_scene_vid = os.path.join(TMP_DIR, f"{job_id}_scene_{i}_video.mp4")
            # render video for scene
            if asset and os.path.exists(asset):
                ext = os.path.splitext(asset)[1].lower()
                if ext in [".mp4", ".mov", ".webm", ".mkv", ".avi"]:
                    # trim/loop clip
                    clip_trim_or_loop(asset, tmp_scene_vid, duration=int(max(4, len(sc["text"].split())/2.5)), resolution=resolution, fps=fps)
                else:
                    image_to_video(asset, tmp_scene_vid, duration=int(max(4, len(sc["text"].split())/2.5)), resolution=resolution, fps=fps)
            else:
                # make text slide image + video
                # render a text image and make a video from it
                from PIL import Image, ImageDraw, ImageFont
                w, h = map(int, resolution.split("x"))
                img = Image.new("RGB", (w,h), color=(20,20,30))
                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 40)
                except:
                    font = ImageFont.load_default()
                # simple wrap
                text = sc["text"]
                draw.text((50, h//2 - 50), text[:200], font=font, fill=(230,230,230))
                tmp_img = os.path.join(TMP_DIR, f"{job_id}_scene_{i}_img.png")
                img.save(tmp_img)
                image_to_video(tmp_img, tmp_scene_vid, duration=int(max(4, len(sc["text"].split())/2.5)), resolution=resolution, fps=fps)

            # mux audio for scene
            tmp_scene_audio = scene_audio_files[i]
            tmp_scene_with_audio = os.path.join(TMP_DIR, f"{job_id}_scene_{i}_final.mp4")
            mux_video_with_audio(tmp_scene_vid, tmp_scene_audio, tmp_scene_with_audio)

            scene_video_files.append(tmp_scene_with_audio)
            JOB_STORE[job_id]["progress"] = 40 + int(30 * (i+1)/max(1,len(scenes)))

        # Concatenate final scenes
        JOB_STORE[job_id]["progress"] = 75
        out_file = os.path.join(OUTPUT_DIR, f"{title}_{job_id}.mp4")
        concat_videos(scene_video_files, out_file)

        JOB_STORE[job_id]["progress"] = 88

        # Generate preview (low-res short sample)
        preview_path = os.path.join(OUTPUT_DIR, f"{title}_{job_id}_preview.mp4")
        generate_preview(out_file, preview_path, preview_seconds=15, res="640x360")
        JOB_STORE[job_id]["preview"] = preview_path

        JOB_STORE[job_id]["file"] = out_file
        JOB_STORE[job_id]["progress"] = 100
        JOB_STORE[job_id]["status"] = "done"

        # cleanup temp files (optional)
        # shutil.rmtree(TMP_DIR)  # do not remove whole tmp dir; remove per-job files if desired

    except Exception as e:
        JOB_STORE[job_id]["status"] = "error"
        JOB_STORE[job_id]["error"] = str(e)
        JOB_STORE[job_id]["progress"] = 0

# --- API endpoints ---

@app.post("/generate_movie")
async def generate_movie_endpoint(
    script: str = Form(...),
    title: Optional[str] = Form("movie"),
    images: Optional[List[UploadFile]] = File(None),
    clips: Optional[List[UploadFile]] = File(None),
    resolution: Optional[str] = Form("1920x1080"),
    fps: Optional[int] = Form(24),
):
    """
    Accepts:
      - script: multi-character script lines like "ALICE: Hi\nBOB: Hello"
      - images[] and clips[] optional
    Returns job_id immediately. Frontend should poll /status/{job_id}.
    """
    job_id = str(uuid.uuid4())
    JOB_STORE[job_id] = {"status": "pending", "progress": 0, "file": None, "preview": None}

    # save uploaded assets to tmp job folder
    job_tmp = os.path.join(TMP_DIR, job_id)
    os.makedirs(job_tmp, exist_ok=True)
    saved_images, saved_clips = [], []

    if images:
        for f in images:
            path = os.path.join(job_tmp, f.filename)
            with open(path, "wb") as fh:
                fh.write(await f.read())
            saved_images.append(path)

    if clips:
        for f in clips:
            path = os.path.join(job_tmp, f.filename)
            with open(path, "wb") as fh:
                fh.write(await f.read())
            saved_clips.append(path)

    # start background worker
    JOB_STORE[job_id]["status"] = "running"
    JOB_STORE[job_id]["progress"] = 1
    asyncio.create_task(worker_render_movie(job_id, script, saved_images, saved_clips, title, resolution=resolution, fps=fps))

    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def status_endpoint(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "job not found"})
    return job

@app.get("/download/{job_id}")
async def download_endpoint(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job or not job.get("file"):
        return JSONResponse(status_code=404, content={"error": "file not ready"})
    filepath = job["file"]
    return FileResponse(filepath, media_type="video/mp4", filename=os.path.basename(filepath))

@app.get("/preview/{job_id}")
async def preview_endpoint(job_id: str):
    job = JOB_STORE.get(job_id)
    if not job or not job.get("preview"):
        return JSONResponse(status_code=404, content={"error": "preview not ready"})
    return FileResponse(job["preview"], media_type="video/mp4", filename=os.path.basename(job["preview"]))
