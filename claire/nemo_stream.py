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

TODO: stamp each emitted delta with a wall-clock timestamp and append to
~/transcripts/live.txt the way claire.py does. For now, raw text -> stdout.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from nemotron_asr_mlx import from_pretrained
from nemotron_asr_mlx.audio import load_audio
import mlx.core as mx
import sys

model = from_pretrained("dboris/nemotron-asr-mlx")

audio = load_audio(sys.argv[1])  # mono float32 @ 16 kHz via ffmpeg

chunk_ms = 1120                   # one of: 80, 160, 560, 1120 — 1120 = max accuracy
chunk_samples = 16000 * chunk_ms // 1000

session = model.create_stream(chunk_ms=chunk_ms)

for start in range(0, len(audio), chunk_samples):
    chunk = mx.array(audio[start:start + chunk_samples])
    event = session.push(chunk)
    print(event.text_delta, end="", flush=True)

session.flush()
print()
