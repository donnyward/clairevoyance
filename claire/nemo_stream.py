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

TODO: append each emitted line to ~/transcripts/live.txt the way claire.py does.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

import collections, datetime, sys, threading
import numpy as np
import sounddevice as sd
import mlx.core as mx
from nemotron_asr_mlx import from_pretrained

SAMPLE_RATE   = 16000
CHUNK_MS      = 1120                              # one of: 80, 160, 560, 1120 — 1120 = max accuracy
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000    # 17,920 samples

print("Loading nemotron-asr-mlx...", flush=True)
model   = from_pretrained("dboris/nemotron-asr-mlx")
session = model.create_stream(chunk_ms=CHUNK_MS)
print(f"Streaming from microphone (chunk={CHUNK_MS}ms, Ctrl+C to stop)", flush=True)

buffer   = collections.deque()
buf_lock = threading.Lock()

def emit(event):
    text = event.text_delta.strip()
    if not text:
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {text}", flush=True)

def audio_cb(indata, frames, time, status):
    if status:
        print(status, file=sys.stderr, flush=True)
    with buf_lock:
        buffer.append(indata[:, 0].copy())

accumulated = np.array([], dtype=np.float32)

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=audio_cb):
        while True:
            with buf_lock:
                while buffer:
                    accumulated = np.concatenate([accumulated, buffer.popleft()])
            while len(accumulated) >= CHUNK_SAMPLES:
                chunk_data, accumulated = accumulated[:CHUNK_SAMPLES], accumulated[CHUNK_SAMPLES:]
                emit(session.push(mx.array(chunk_data)))
            sd.sleep(50)
except KeyboardInterrupt:
    pass

with buf_lock:
    while buffer:
        accumulated = np.concatenate([accumulated, buffer.popleft()])
if len(accumulated) > 0:
    emit(session.push(mx.array(accumulated)))
session.flush()
