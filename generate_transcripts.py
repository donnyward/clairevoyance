#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyannote.audio>=3.1",
#   "soundfile",
#   "numpy",
#   "torch",
# ]
# ///
"""Generate speaker-named transcripts from Whisper JSON diarization files.

On top of today's name-prompt flow, this also offers to save a voice embedding
for each named speaker (pyannote/wespeaker) into a local SQLite DB. Embeddings
are only saved when you explicitly type 'y' at the save prompt — so you
manually curate which segments get enrolled.
"""
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile


HF_TOKEN = "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"
DB_FILENAME = "speakers.db"
MIN_SAVE_SECONDS = 3.0
MAX_SAVE_SECONDS = 20.0  # cap enrollment clip length; past ~20s quality plateaus
                         # and the odds of diarization bleeding in other speakers climb
EMBED_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

SCHEMA = """
CREATE TABLE IF NOT EXISTS speakers (
  id   INTEGER PRIMARY KEY,
  name TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
  id           INTEGER PRIMARY KEY,
  speaker_id   INTEGER NOT NULL REFERENCES speakers(id),
  vec          BLOB    NOT NULL,
  source_file  TEXT    NOT NULL,
  start_time   REAL    NOT NULL,
  end_time     REAL    NOT NULL,
  created_at   TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (speaker_id, source_file, start_time)
);
"""


def format_timestamp(seconds):
    """Convert seconds (float) to MM:SS string."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def flatten_words(segments):
    """Extract all words with speaker labels and timestamps from segments."""
    words = []
    for seg in segments:
        seg_speaker = seg.get("speaker")
        for w in seg.get("words", []):
            sp = w.get("speaker", seg_speaker)
            start = w.get("start", 0.0)
            end = w.get("end", start)
            words.append((sp, w.get("word", ""), start, end))
    return words


def group_words_by_speaker(words):
    """Group consecutive words by speaker.

    Returns list of (speaker, text, start_time, end_time) tuples.
    """
    groups = []
    current_speaker = None
    current_words = []
    current_start = 0.0
    current_end = 0.0
    for sp, word, start, end in words:
        if not word:
            continue
        if sp == current_speaker:
            current_words.append(word)
            current_end = end
        else:
            if current_speaker is not None and current_words:
                groups.append((current_speaker, " ".join(current_words),
                               current_start, current_end))
            current_speaker = sp
            current_words = [word]
            current_start = start
            current_end = end
    if current_speaker is not None and current_words:
        groups.append((current_speaker, " ".join(current_words),
                       current_start, current_end))
    return groups


def find_longest_line(word_groups, speaker):
    """Find the longest consecutive run of words for a speaker.

    Returns (text, start_time, end_time).
    """
    best_text = ""
    best_start = 0.0
    best_end = 0.0
    for sp, text, start, end in word_groups:
        if sp == speaker and len(text) > len(best_text):
            best_text = text
            best_start = start
            best_end = end
    return best_text, best_start, best_end


def load_embedding_model(hf_token):
    """Load pyannote embedding model. Returns Inference callable, or None on failure."""
    try:
        from pyannote.audio import Inference, Model
        model = Model.from_pretrained(EMBED_MODEL, use_auth_token=hf_token)
        return Inference(model, window="whole")
    except Exception as e:
        print(f"WARNING: embedding model failed to load: {e}", file=sys.stderr)
        print("         transcription will continue without embedding saves.",
              file=sys.stderr)
        return None


def init_db(path):
    """Open SQLite DB at path and ensure schema exists."""
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def get_or_create_speaker(conn, name):
    cur = conn.execute("SELECT id FROM speakers WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur = conn.execute("INSERT INTO speakers (name) VALUES (?)", (name,))
    return cur.lastrowid


def save_embedding(conn, speaker_id, vec, source_file, start, end):
    import numpy as np
    blob = np.asarray(vec, dtype=np.float32).tobytes()
    conn.execute(
        "INSERT OR IGNORE INTO embeddings "
        "(speaker_id, vec, source_file, start_time, end_time) "
        "VALUES (?, ?, ?, ?, ?)",
        (speaker_id, blob, source_file, start, end),
    )


def play_segment(wav, sr):
    """Write wav to a temp file and start afplay in the background. Returns
    (proc, path); pass both to stop_playback to clean up."""
    import soundfile as sf
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, wav, sr)
    proc = subprocess.Popen(["afplay", path])
    return proc, path


def stop_playback(proc, path):
    """Kill afplay if still running, then unlink the temp file."""
    try:
        proc.terminate()
        proc.wait(timeout=1)
    except Exception:
        pass
    try:
        os.unlink(path)
    except OSError:
        pass


def extract_audio(audio_path, start, end):
    """Pull a 16kHz mono WAV segment out of audio_path via ffmpeg. Returns (wav, sr)."""
    import numpy as np
    import soundfile as sf
    cmd = [
        "ffmpeg", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
        "-i", audio_path,
        "-ar", "16000", "-ac", "1", "-f", "wav",
        "-loglevel", "error", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    wav, sr = sf.read(io.BytesIO(result.stdout))
    return wav.astype(np.float32), sr


def compute_embedding(inference, wav, sr):
    """Run pyannote on a waveform. Returns L2-normalized float32 vector."""
    import numpy as np
    import torch
    waveform = torch.from_numpy(wav).unsqueeze(0)  # (1, num_samples)
    vec = inference({"waveform": waveform, "sample_rate": sr})
    vec = np.asarray(vec).flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def process_json(json_path, txt_path, audio_path, embed_inference, conn):
    """Parse a whisper JSON, prompt for speaker names (+ optional embedding save),
    and write a named transcript. Returns True on success, False if user skipped."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        print(f"  No segments found, skipping.")
        return True
    words = flatten_words(segments)
    if not words:
        print(f"  No words found, skipping.")
        return True
    groups = group_words_by_speaker(words)
    seen = set()
    speakers = []
    for sp, _, _, _ in groups:
        if sp and sp not in seen:
            seen.add(sp)
            speakers.append(sp)
    if not speakers:
        print(f"  No speaker labels found, skipping.")
        return True

    print(f"\n--- {os.path.basename(json_path)} ---\n")
    name_map = {}
    for sp in speakers:
        _, start, end = find_longest_line(groups, sp)
        duration = end - start
        embed_end = min(end, start + MAX_SAVE_SECONDS)
        embed_duration = embed_end - start
        # Build the ID hint from ONLY the words attributed to this speaker inside
        # the enrollment window — user uses this to spot mis-attributed one-word
        # utterances (diarization bleed) before deciding to save.
        enrolled_text = " ".join(
            w for sp2, w, ws, _ in words
            if sp2 == sp and start <= ws < embed_end and w
        )
        ts = format_timestamp(start)
        if embed_duration < duration:
            ts_end = format_timestamp(embed_end)
            print(f'  {sp} (longest run: {duration:.1f}s -> enrolling '
                  f'{embed_duration:.1f}s from {ts} to {ts_end}):')
        else:
            print(f'  {sp} (longest line at {ts}, {duration:.1f}s):')
        print(f'   "{enrolled_text}"')

        # Extract the enrollment clip once and play it during BOTH the naming
        # prompt and (if applicable) the save prompt. Same bytes the embedder
        # will see — what you hear is what gets stored.
        wav, sr = None, None
        if os.path.exists(audio_path):
            try:
                wav, sr = extract_audio(audio_path, start, embed_end)
            except Exception as e:
                print(f"  !! audio extract failed: {e}")

        proc, tmppath = None, None
        try:
            if wav is not None:
                proc, tmppath = play_segment(wav, sr)
            replay_hint = ", 'r' to replay" if wav is not None else ""
            while True:
                name = input(f"Who is {sp}? ('s' to skip{replay_hint}) ").strip()
                if name.lower() == "r" and wav is not None:
                    stop_playback(proc, tmppath)
                    proc, tmppath = play_segment(wav, sr)
                    continue
                break
        finally:
            if proc is not None:
                stop_playback(proc, tmppath)
                proc, tmppath = None, None

        if name.lower() in ("s", "skip"):
            # Undo any embeddings committed from this file earlier in the session
            # so "skip" still means "this file's diarization is bad, discard it".
            if conn is not None:
                conn.execute(
                    "DELETE FROM embeddings WHERE source_file = ?", (audio_path,))
                conn.commit()
            print(f"  Skipping {os.path.basename(json_path)} (bad diarization).")
            return False
        if not name:
            name = sp
        name_map[sp] = name

        can_save = (
            embed_inference is not None
            and conn is not None
            and name != sp                          # not a real person name
            and embed_duration >= MIN_SAVE_SECONDS  # too short for a reliable embedding
            and wav is not None
        )
        if can_save:
            try:
                proc, tmppath = play_segment(wav, sr)
                while True:
                    ans = input("Save embedding? [y/N/r=replay] ").strip().lower()
                    if ans != "r":
                        break
                    stop_playback(proc, tmppath)
                    proc, tmppath = play_segment(wav, sr)
                if ans in ("y", "yes"):
                    vec = compute_embedding(embed_inference, wav, sr)
                    speaker_id = get_or_create_speaker(conn, name)
                    save_embedding(conn, speaker_id, vec, audio_path, start, embed_end)
                    conn.commit()
                    print(f"  -> saved ({embed_duration:.1f}s from "
                          f"{os.path.basename(audio_path)} -> {name})")
            except Exception as e:
                print(f"  !! Failed to save embedding: {e}")
            finally:
                if proc is not None:
                    stop_playback(proc, tmppath)
        print()

    # Merge consecutive groups from the same speaker into blocks
    blocks = []
    current_speaker = None
    current_lines = []
    for sp, text, _, _ in groups:
        if sp == current_speaker:
            current_lines.append(text)
        else:
            if current_speaker is not None:
                blocks.append((current_speaker, current_lines))
            current_speaker = sp
            current_lines = [text]
    if current_speaker is not None:
        blocks.append((current_speaker, current_lines))
    # Write transcript
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, (sp, lines) in enumerate(blocks):
            name = name_map.get(sp, sp)
            f.write(f"{name}:\n")
            f.write("\n".join(lines))
            f.write("\n")
            if i < len(blocks) - 1:
                f.write("\n")
    print(f"  Wrote {txt_path}")
    return True


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    embed_inference = load_embedding_model(HF_TOKEN)
    conn = None
    if embed_inference is not None:
        try:
            conn = init_db(os.path.join(script_dir, DB_FILENAME))
        except Exception as e:
            print(f"WARNING: could not open {DB_FILENAME}: {e}", file=sys.stderr)
            print("         transcription will continue without embedding saves.",
                  file=sys.stderr)

    count_processed = 0
    count_skipped_exists = 0
    count_skipped_no_json = 0
    count_skipped_diarization = 0
    for root, _dirs, files in os.walk(script_dir):
        for fname in sorted(files):
            if not fname.endswith(".m4a"):
                continue
            base = fname[:-4]  # strip .m4a
            json_path = os.path.join(root, base + "_whisper.json")
            txt_path = os.path.join(root, base + "_whisper.txt")
            audio_path = os.path.join(root, fname)
            if os.path.exists(txt_path):
                print(f"Skip (txt exists): {txt_path}")
                count_skipped_exists += 1
                continue
            if not os.path.exists(json_path):
                print(f"Skip (no json): {fname}")
                count_skipped_no_json += 1
                continue
            if process_json(json_path, txt_path, audio_path, embed_inference, conn):
                count_processed += 1
            else:
                count_skipped_diarization += 1

    if conn is not None:
        conn.close()

    print(f"\nDone. Processed: {count_processed}; "
          f"Skipped (txt exists): {count_skipped_exists}; "
          f"Skipped (no json): {count_skipped_no_json}; "
          f"Skipped (bad diarization): {count_skipped_diarization}")


if __name__ == "__main__":
    main()
