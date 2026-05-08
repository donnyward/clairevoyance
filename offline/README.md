# Offline

Pipeline for archiving and transcribing recorded audio (m4a/mp3/flac) on Apple Silicon. Stages new files from `/tmp` into a date-organized archive, transcribes with whisper-mlx, and resolves speaker names via pyannote embeddings.

## Pipeline

Typical flow for new recordings:

1. Drop audio files into `/tmp` (exported from Voice Memos, copied from other recorders, etc.). Filenames must start with `YYYYMMDD_`.
2. `./ingest.py <archive_root>` — moves them into `<archive_root>/YYYY-MM-DD/` and extracts any embedded iOS Voice Memo transcripts.
3. `./transcribe.sh <archive_root>` — generates `*_whisper.json` for any audio missing one.
4. `generate_transcripts.py <archive_root>` — produces named, speaker-resolved text transcripts from the whisper JSON.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Dependencies and the Python interpreter are managed via `pyproject.toml` + `uv.lock` — `uv run` creates `.venv/` and installs everything on first invocation. To update after editing `pyproject.toml`, run `uv sync`.

You'll also need `ffmpeg` on `PATH` (the shell scripts shell out to it for resampling).

A Hugging Face token with the pyannote terms accepted is required; the token is currently hardcoded in the scripts.

### Pre-download models for offline use

The shell scripts and `generate_transcripts.py` set `HF_HUB_OFFLINE=1`, so all
models must already be cached locally. Run the downloader in `../download/`
once with internet access to seed the caches:

```bash
cd ../download && HF_TOKEN=hf_xxx uv run download_models.py
```

Subsequent runs work without network as long as package versions in `uv.lock`
don't change.

## Usage

### Ingest

`ingest.py` moves audio files (`.m4a`, `.mp3`, `.flac`) from `/tmp` into a date-organized folder under the archive root, and extracts any embedded iOS Voice Memo transcripts. Stdlib-only, no `uv` needed.

```bash
/path/to/offline/ingest.py /path/to/archive
```

Phase 1 — for each audio file in `/tmp` whose name starts with `YYYYMMDD_`:
- Moves it to `<archive>/YYYY-MM-DD/`.
- Also moves any `/tmp/*.txt` whose basename starts with the audio file's stem (e.g. `20260101_120000_notes.txt` follows `20260101_120000.m4a`).
- Files without a date prefix are warned and left in `/tmp`.

Phase 2 — for each `.m4a` in the archive without a `_voicememo.txt` sidecar, extracts the iOS Voice Memo transcript from the `moov/trak/udta/tsrp` atom and writes it to `<basename>_voicememo.txt`. Files lacking the atom (e.g. Samsung Recorder m4a) are logged but not flagged.

A summary at the end reports moved / skipped / extracted counts for both phases.

### Transcribe

`transcribe.sh` searches the current directory by default, or pass a starting directory as the first argument. It discovers `*.m4a`, `*.mp3`, `*.flac` recursively.

```bash
/path/to/offline/transcribe.sh                       # search cwd
/path/to/offline/transcribe.sh /path/to/recordings   # search a specific directory
```

The script:
1. Lists all audio files and skips any with an existing `*_whisper.json` next to them.
2. Prompts per file: Enter for auto-detect speakers, a number to fix the speaker count, or `s` to skip.
3. Runs `ffmpeg` → `whispermlx --diarize` and writes `<basename>_whisper.json` next to each source file.

### Generate named transcripts from `*_whisper.json`

Walks the current directory (or the directory passed as the first arg) for audio files with adjacent `*_whisper.json` and writes named transcripts:

```bash
# Process the cwd
uv run --project /path/to/offline /path/to/offline/generate_transcripts.py

# Or point at a specific directory
uv run --project /path/to/offline /path/to/offline/generate_transcripts.py /path/to/recordings
```

Prompts for a name per `SPEAKER_XX`, suggests ranked candidates from the local `speakers.db` when prior enrollments exist, and optionally enrolls new embeddings. The DB stays anchored at `offline/speakers.db` regardless of which directory is processed.

### Normalize recording filenames

```bash
uv run --project /path/to/offline /path/to/offline/rename_dry_run.py            # dry run
uv run --project /path/to/offline /path/to/offline/rename_dry_run.py --execute  # apply
```

Renames files within each recording folder so the basename matches the folder name, preserving known suffixes.
