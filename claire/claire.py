#!/usr/bin/env -S uv run
"""claire.py - real-time transcription with speaker segmentation."""
import os, datetime, collections, threading, queue
import numpy as np
import sounddevice as sd
import mlx_whisper
import torch
from pyannote.audio import Model as PyanModel
from resemblyzer import VoiceEncoder, preprocess_wav

os.environ["HF_HUB_OFFLINE"] = "1"

LIVE_FILE      = os.path.expanduser("~/transcripts/live.txt")
WHISPER_DIR    = os.path.expanduser("~/.cache/mlx-whisper/large-v3-turbo")
SEG_DIR        = os.path.expanduser("~/.cache/pyannote/segmentation-3.0")
SAMPLE_RATE    = 16000
CHUNK_SECS     = 10
SPEAKER_THRESH = 0.82

os.makedirs(os.path.dirname(LIVE_FILE), exist_ok=True)

# --- Models (loaded once at startup) ---
print(f"Loading resemblyzer VoiceEncoder...", flush=True)
encoder    = VoiceEncoder()
print(f"Loading pyannote segmentation: {SEG_DIR}", flush=True)
seg_model  = PyanModel.from_pretrained(SEG_DIR)
seg_model.eval()
print(f"Loading silero-vad...", flush=True)
vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", verbose=False,
                              trust_repo=True)
print("All models loaded.", flush=True)

# --- Session speaker state ---
session_speakers: list[np.ndarray] = []
spk_lock = threading.Lock()

def get_speaker_label(audio_segment: np.ndarray) -> str:
    wav = preprocess_wav(audio_segment, source_sr=SAMPLE_RATE)
    if len(wav) < 3200:
        return "Speaker ?"
    emb = encoder.embed_utterance(wav)
    with spk_lock:
        for i, ref in enumerate(session_speakers):
            if float(np.dot(emb, ref)) >= SPEAKER_THRESH:
                return f"Speaker {i + 1}"
        session_speakers.append(emb)
        return f"Speaker {len(session_speakers)}"

def is_speech(audio: np.ndarray, threshold: float = 0.1) -> bool:
    window = 512
    positives = 0
    total = 0
    vad_model.reset_states()
    with torch.no_grad():
        for i in range(0, len(audio) - window + 1, window):
            t = torch.from_numpy(audio[i:i + window]).float()
            if vad_model(t, SAMPLE_RATE).item() > 0.5:
                positives += 1
            total += 1
    return total > 0 and (positives / total) >= threshold

def segment_speakers(audio: np.ndarray):
    waveform = torch.from_numpy(audio).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = seg_model(waveform)
        activations = torch.sigmoid(output[0]).numpy()  # (num_frames, num_speakers)
    num_frames, num_speakers = activations.shape
    frame_duration = len(audio) / num_frames

    for spk_idx in range(num_speakers):
        col = activations[:, spk_idx]
        print(f"  [seg] speaker {spk_idx}: max={col.max():.3f} mean={col.mean():.3f}", flush=True)

    segments = []
    for spk_idx in range(num_speakers):
        active = activations[:, spk_idx] > 0.5
        in_segment = False
        start = 0
        for f in range(num_frames):
            if active[f] and not in_segment:
                start = f
                in_segment = True
            elif not active[f] and in_segment:
                s = int(start * frame_duration)
                e = int(f * frame_duration)
                if e - s > 1600:
                    segments.append((s, e, spk_idx))
                in_segment = False
        if in_segment:
            s = int(start * frame_duration)
            segments.append((s, len(audio), spk_idx))
    segments.sort(key=lambda x: x[0])
    return segments

import re, unicodedata

def is_hallucination(text: str) -> bool:
    latin_count = sum(1 for c in text if unicodedata.category(c).startswith("L")
                                          and ord(c) < 0x0250)
    letter_count = sum(1 for c in text if unicodedata.category(c).startswith("L"))
    if letter_count > 5 and latin_count / letter_count < 0.5:
        return True
    words = text.split()
    if len(words) >= 6:
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < 0.2:
            return True
    return False

def transcribe_segment(seg_audio: np.ndarray) -> None:
    speaker = get_speaker_label(seg_audio)
    result = mlx_whisper.transcribe(
        seg_audio, path_or_hf_repo=WHISPER_DIR, language="en")
    text = result.get("text", "").strip()
    if not text:
        return
    if is_hallucination(text):
        print(f"  [filtered] {text[:80]}..." if len(text) > 80 else f"  [filtered] {text}", flush=True)
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{speaker}] {text}"
    print(line, flush=True)
    with open(LIVE_FILE, "a") as f:
        f.write(line + "\n")

def process_chunk(chunk: np.ndarray) -> None:
    print(f"\n{'-'*50}", flush=True)
    print(f"  [chunk] processing {len(chunk)} samples...", flush=True)
    if not is_speech(chunk):
        print("  [chunk] no speech detected, skipping", flush=True)
        return
    print("  [chunk] speech detected, segmenting...", flush=True)
    segments = segment_speakers(chunk)
    if not segments:
        print("  [chunk] no speaker segments found, transcribing whole chunk...", flush=True)
        transcribe_segment(chunk)
    else:
        print(f"  [chunk] {len(segments)} segment(s), transcribing...", flush=True)
        for start, end, _ in segments:
            transcribe_segment(chunk[start:end])

# --- Single worker thread for sequential model access ---
work_queue: queue.Queue[np.ndarray | None] = queue.Queue()

def worker():
    while True:
        chunk = work_queue.get()
        if chunk is None:
            break
        try:
            process_chunk(chunk)
        except Exception as e:
            print(f"[error] {e}", flush=True)

worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

# --- Audio capture loop ---
buffer    = collections.deque()
buf_lock  = threading.Lock()

def audio_cb(indata, frames, time, status):
    with buf_lock:
        buffer.append(indata[:, 0].copy())

print(f"Transcribing → {LIVE_FILE} (Ctrl+C to stop)")
chunk_samples = SAMPLE_RATE * CHUNK_SECS
accumulated   = np.array([], dtype=np.float32)

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_cb):
    try:
        while True:
            with buf_lock:
                while buffer:
                    accumulated = np.concatenate([accumulated, buffer.popleft()])
            if len(accumulated) >= chunk_samples:
                chunk, accumulated = accumulated[:chunk_samples], accumulated[chunk_samples:]
                work_queue.put(chunk.copy())
            else:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\nStopping...")
        work_queue.put(None)
        worker_thread.join(timeout=10)
