# Claire

Real-time speech transcription with speaker diarization, running locally on Apple Silicon via MLX. All audio stays in memory — nothing is written to disk except the text transcript.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies and the Python interpreter are managed via `pyproject.toml` + `uv.lock` — `uv run` creates `.venv/` and installs everything on first invocation.

```bash
# 1. Get a Hugging Face token and accept the pyannote terms:
#    https://huggingface.co/settings/tokens
#    https://huggingface.co/pyannote/segmentation-3.0

# 2. Download all models (~800MB for whisper large-v3-turbo + smaller models).
#    The downloader lives in ../download/ and seeds caches for both subprojects.
cd ../download && HF_TOKEN=hf_xxx uv run download_models.py && cd -
```

To update dependencies after editing `pyproject.toml`, run `uv sync` (or just `uv run claire.py` — sync happens implicitly).

## Usage

### Start a session

```bash
uv run claire.py
```

Listens to your microphone, detects speech, identifies speakers, and transcribes in real time.
Output is printed to the console and appended to `~/transcripts/live.txt`.

### Stop a session

Press `Ctrl+C`.

### Archive and start a new session

```bash
# Move the transcript somewhere meaningful
mv ~/transcripts/live.txt ~/transcripts/2026-04-29-hackathon.txt

# Start a new session - live.py creates a fresh live.txt automatically
uv run claire.py
```

No need to reset any cursor files or clean up state. The inject cursor (`~/transcripts/.inject_cursor`) auto-resets when it detects the file was replaced.

Speaker labels (Speaker 1, Speaker 2, etc.) reset each session since they're tracked in memory.

## Claude Code integration

A `UserPromptSubmit` hook injects new transcript lines into every Claude prompt automatically. When `claire.py` is running, Claude sees the room conversation incrementally. When it's not running, the hook is silent.

To set up on a new machine:

1. **Add the hook** — merge the contents of `hook-config.json` into your `~/.claude/settings.json`, updating the path to where you cloned `inject_transcript.py`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/inject_transcript.py"
          }
        ]
      }
    ]
  }
}
```

2. **Add the CLAUDE.md snippet** — append the contents of `claude-md-snippet.md` to your `~/.claude/CLAUDE.md`. This tells Claude what the `<live-transcript>` tags mean and how to interpret imperfect transcriptions.

## How it works

- **VAD** - Silero VAD filters silence so only speech chunks are processed
- **Segmentation** - pyannote segmentation-3.0 separates overlapping speakers within each chunk
- **Speaker ID** - resemblyzer assigns consistent speaker labels across the session
- **Transcription** - mlx-whisper (large-v3-turbo) runs Whisper natively on Apple Silicon
