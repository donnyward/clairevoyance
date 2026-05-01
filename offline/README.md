# Offline

Batch transcription of recorded audio files (m4a/mp3/flac) with WhisperX, plus speaker-name resolution and enrollment via pyannote embeddings.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies and the Python interpreter are managed via `pyproject.toml` + `uv.lock` — `uv run` creates `.venv/` and installs everything on first invocation. To update after editing `pyproject.toml`, run `uv sync`.

You'll also need `ffmpeg` on `PATH` (the shell scripts shell out to it for resampling).

A Hugging Face token with the pyannote terms accepted is required; the token is currently hardcoded in the scripts.

### Pre-download models for offline use

The shell scripts and `generate_transcripts.py` set `HF_HUB_OFFLINE=1`, so all
models must already be cached locally. Run the top-level downloader once with
internet access to seed the caches:

```bash
HF_TOKEN=hf_xxx uv run ../download_models.py
```

Subsequent runs work without network as long as package versions in `uv.lock`
don't change.

## Usage

Run the scripts from the directory containing the audio files. They discover `*.m4a`, `*.mp3`, `*.flac` recursively under `.`.

### Transcribe (CPU, Apple Silicon / generic)

```bash
/path/to/offline/transcribe.sh
```

### Transcribe (CUDA GPU)

```bash
/path/to/offline/transcribe_gpu.sh
```

Both scripts:
1. List all audio files and skip any with an existing `*_whisper.json` next to them.
2. Prompt per file: Enter for auto-detect speakers, a number to fix the speaker count, or `skip`.
3. Run `ffmpeg` → `whisperx --diarize` and write `<basename>_whisper.json` next to each source file.

### Generate named transcripts from `*_whisper.json`

```bash
uv run --project /path/to/offline /path/to/offline/generate_transcripts.py <whisper.json> [...]
```

Prompts for a name per `SPEAKER_XX`, suggests ranked candidates from the local `speakers.db` when prior enrollments exist, and optionally enrolls new embeddings.

### Normalize recording filenames

```bash
uv run --project /path/to/offline /path/to/offline/rename_dry_run.py            # dry run
uv run --project /path/to/offline /path/to/offline/rename_dry_run.py --execute  # apply
```

Renames files within each recording folder so the basename matches the folder name, preserving known suffixes.
