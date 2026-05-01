#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "huggingface-hub",
#   "truststore",
#   "torch",
#   "torchaudio",
#   "whisperx",
# ]
# ///
"""One-time model download for both claire/ and offline/ subprojects.

Run: HF_TOKEN=hf_xxx uv run download_models.py

After this completes, both subprojects can run with HF_HUB_OFFLINE=1:
  - claire/claire.py reads pyannote + mlx-whisper from ~/.cache/{pyannote,mlx-whisper}
  - offline/transcribe*.sh and generate_transcripts.py read whisperx + pyannote
    from the default HF cache (~/.cache/huggingface) and torch caches.
"""
import os
import sys

import truststore
truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_XET"] = "1"


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_token")
        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
    if not token:
        print("ERROR: Set HF_TOKEN=hf_xxx or write the token to ./hf_token")
        print("  Token: https://huggingface.co/settings/tokens")
        print("  Accept terms for each gated repo:")
        print("    https://huggingface.co/pyannote/segmentation-3.0")
        print("    https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("    https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
        sys.exit(1)

    from huggingface_hub import snapshot_download

    # --- claire/: local-path models (claire.py reads them by absolute path) ---
    print("\n[claire] mlx-whisper large-v3-turbo (~800MB) -> ~/.cache/mlx-whisper/...")
    snapshot_download(
        "mlx-community/whisper-large-v3-turbo",
        local_dir=os.path.expanduser("~/.cache/mlx-whisper/large-v3-turbo"),
    )

    print("\n[claire] pyannote/segmentation-3.0 (~5MB) -> ~/.cache/pyannote/...")
    snapshot_download(
        "pyannote/segmentation-3.0",
        local_dir=os.path.expanduser("~/.cache/pyannote/segmentation-3.0"),
        token=token,
    )

    print("\n[claire] silero-vad (~2MB) -> torch hub cache...")
    import torch
    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True, verbose=False)

    # --- offline/: HF cache models (whisperx + generate_transcripts read from cache) ---
    # Use whisperx's own loaders so every cache it touches at runtime is warmed:
    # faster-whisper weights, the torchaudio wav2vec2 bundle, the diarization
    # pipeline (which pulls segmentation-3.0 + wespeaker), and whisperx's VAD.
    import whisperx

    print("\n[offline] whisperx large-v3 (faster-whisper ~3GB, includes VAD ~17MB)...")
    whisperx.load_model("large-v3", device="cpu", compute_type="int8")

    print("\n[offline] whisperx align model (WAV2VEC2_ASR_LARGE_LV60K_960H ~1.2GB)...")
    whisperx.load_align_model(language_code="en", device="cpu")

    print("\n[offline] pyannote diarization pipeline (segmentation + wespeaker)...")
    from whisperx.diarize import DiarizationPipeline
    DiarizationPipeline(token=token, device="cpu")

    print("\nDone. All models cached locally. Set HF_HUB_OFFLINE=1 at runtime.")


if __name__ == "__main__":
    main()
