#!/usr/bin/env -S uv run
"""One-time model download. Run: HF_TOKEN=hf_xxx uv run download_models.py"""
import os, sys, truststore
truststore.inject_into_ssl()
os.environ["HF_HUB_DISABLE_XET"] = "1"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN=hf_xxx")
        print("  Token: https://huggingface.co/settings/tokens")
        print("  Accept terms: https://huggingface.co/pyannote/segmentation-3.0")
        sys.exit(1)

    from huggingface_hub import snapshot_download

    print("1/4 mlx-whisper large-v3-turbo (~800MB)...")
    snapshot_download("mlx-community/whisper-large-v3-turbo",
                      local_dir=os.path.expanduser("~/.cache/mlx-whisper/large-v3-turbo"))

    print("2/4 pyannote/segmentation-3.0 (~5MB)...")
    snapshot_download("pyannote/segmentation-3.0",
                      local_dir=os.path.expanduser("~/.cache/pyannote/segmentation-3.0"),
                      token=token)

    print("3/4 resemblyzer GE2E encoder (~17MB)...")
    from resemblyzer import VoiceEncoder
    VoiceEncoder()

    print("4/4 silero-vad (~2MB)...")
    import torch
    torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)

    print("\nDone. All models cached locally. Set HF_HUB_OFFLINE=1 at runtime.")

if __name__ == "__main__":
    main()
