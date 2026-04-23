#!/usr/bin/env python3
"""Generate speaker-named transcripts from Whisper JSON diarization files."""
import json
import os
import sys


def format_timestamp(seconds):
    """Convert seconds (float) to MM:SS string."""
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def flatten_words(segments):
    """Extract all words with speaker labels from segments.

    Uses word-level speaker if present, otherwise falls back to the
    parent segment's speaker. Returns list of (speaker, word, start) tuples.
    """
    words = []
    for seg in segments:
        seg_speaker = seg.get("speaker")
        for w in seg.get("words", []):
            sp = w.get("speaker", seg_speaker)
            words.append((sp, w.get("word", ""), w.get("start", 0.0)))
    return words


def find_longest_line(word_groups, speaker):
    """Find the longest consecutive run of words for a speaker.

    word_groups is a list of (speaker, text, start_time) blocks.
    """
    best_text = ""
    best_start = 0.0
    for sp, text, start in word_groups:
        if sp == speaker and len(text) > len(best_text):
            best_text = text
            best_start = start
    return best_text, best_start


def group_words_by_speaker(words):
    """Group consecutive words by speaker, returns list of (speaker, text, start_time) tuples."""
    groups = []
    current_speaker = None
    current_words = []
    current_start = 0.0
    for sp, word, start in words:
        if not word:
            continue
        if sp == current_speaker:
            current_words.append(word)
        else:
            if current_speaker is not None and current_words:
                groups.append((current_speaker, " ".join(current_words), current_start))
            current_speaker = sp
            current_words = [word]
            current_start = start
    if current_speaker is not None and current_words:
        groups.append((current_speaker, " ".join(current_words), current_start))
    return groups


def process_json(json_path, txt_path):
    """Parse a whisper JSON and write a speaker-named transcript."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    if not segments:
        print(f"  No segments found, skipping.")
        return True
    # Flatten to word-level speaker assignments
    words = flatten_words(segments)
    if not words:
        print(f"  No words found, skipping.")
        return True
    # Group consecutive words by speaker
    groups = group_words_by_speaker(words)
    # Collect unique speakers in order of appearance
    seen = set()
    speakers = []
    for sp, _, _ in groups:
        if sp and sp not in seen:
            seen.add(sp)
            speakers.append(sp)
    if not speakers:
        print(f"  No speaker labels found, skipping.")
        return True
    # Prompt user for each speaker's name
    print(f"\n--- {os.path.basename(json_path)} ---\n")
    name_map = {}
    for sp in speakers:
        text, start = find_longest_line(groups, sp)
        ts = format_timestamp(start)
        print(f'  {sp} (longest line at {ts}):')
        print(f'   "{text}"')
        name = input(f"Who is {sp}? (type 'skip' to skip this file) ").strip()
        if name.lower() == "skip":
            print(f"  Skipping {os.path.basename(json_path)} (bad diarization).")
            return False
        if not name:
            name = sp
        name_map[sp] = name
        print()
    # Merge consecutive groups from the same speaker into blocks
    blocks = []
    current_speaker = None
    current_lines = []
    for sp, text, _ in groups:
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
            if os.path.exists(txt_path):
                print(f"Skip (txt exists): {txt_path}")
                count_skipped_exists += 1
                continue
            if not os.path.exists(json_path):
                print(f"Skip (no json): {fname}")
                count_skipped_no_json += 1
                continue
            if process_json(json_path, txt_path):
                count_processed += 1
            else:
                count_skipped_diarization += 1
    print(f"\nDone. Processed: {count_processed}; "
          f"Skipped (txt exists): {count_skipped_exists}; "
          f"Skipped (no json): {count_skipped_no_json}; "
          f"Skipped (bad diarization): {count_skipped_diarization}")


if __name__ == "__main__":
    main()
