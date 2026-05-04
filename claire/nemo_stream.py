#!/usr/bin/env -S uv run
"""Streaming transcription experiment using NVIDIA Nemotron-ASR (MLX port).

Why this and not the alternatives:

- Tried parakeet-mlx (TDT 0.6b v2 and v3). Batch transcribe() OOMs on long files
  because Conformer self-attention is O(T^2). The parakeet-mlx CLI works around
  this with overlap-chunked batch inference, but boundary artifacts hurt quality
  and the streaming API (transcribe_stream) emits very few finalized tokens at
  high depth — bad fit for appending to a live transcript.

- Tried nemotron-asr-mlx batch mode. Same OOM problem on long inputs.

- Settled on nemotron-asr-mlx in cache-aware streaming mode at chunk_ms=1120.
  Cache-aware Conformer carries encoder KV state and decoder LSTM state across
  chunks via fixed-size ring buffers, so memory and per-chunk latency stay
  constant regardless of how long the mic has been open. No O(T^2) blow-up,
  no boundary artifacts (state flows across chunks at the layer level), and
  1120 ms is the chunk size NVIDIA used for their reference WER benchmark.

This is the right architecture for the live-mic use case: a multi-hour seminar
costs the same per chunk at hour 9 as it did at second 5.

Streaming gives up beam search / ILM subtraction / KenLM fusion (those only
exist in batch mode), which is ~0.3% absolute WER. Worth it for constant
performance on unbounded input.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import datetime, queue, sys
import numpy as np
import sounddevice as sd
import mlx.core as mx
from nemotron_asr_mlx import from_pretrained

LIVE_FILE     = os.path.expanduser("~/transcripts/live.txt")
SAMPLE_RATE   = 16000
CHUNK_MS      = 1120                              # one of: 80, 160, 560, 1120 — 1120 = max accuracy
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000    # 17,920 samples

os.makedirs(os.path.dirname(LIVE_FILE), exist_ok=True)

print("Loading nemotron-asr-mlx...", flush=True)
model   = from_pretrained("dboris/nemotron-asr-mlx")
session = model.create_stream(chunk_ms=CHUNK_MS)

# JIT-compile MLX kernels on a silent dummy chunk so the first real chunk of
# speech doesn't sit in the audio queue waiting for compilation. reset()
# discards the dummy's cache state so the encoder sees the live audio fresh.
print("Warming up...", flush=True)
session.push(mx.array(np.zeros(CHUNK_SAMPLES, dtype=np.float32)))
session.reset()

print(f"Transcribing → {LIVE_FILE} (chunk={CHUNK_MS}ms, Ctrl+C to stop)", flush=True)

audio_q: queue.Queue = queue.Queue()

def emit(event):
    text = event.text_delta.strip()
    if not text:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {text}"
    print(line, flush=True)
    with open(LIVE_FILE, "a") as f:
        f.write(line + "\n")

def audio_cb(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr, flush=True)
    audio_q.put(indata[:, 0].copy())

try:
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,            # one full chunk per callback — no accumulator needed
        callback=audio_cb,
    ):
        while True:
            emit(session.push(mx.array(audio_q.get())))
except KeyboardInterrupt:
    pass

while not audio_q.empty():
    emit(session.push(mx.array(audio_q.get_nowait())))
session.flush()
