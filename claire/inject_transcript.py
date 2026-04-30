#!/usr/bin/env python3
"""Injects new transcript lines into Claude Code context.

Used as a UserPromptSubmit hook - stdout becomes part of Claude's context.
Tracks a byte-offset cursor so each prompt only sees new lines since the last one.
Silent (no output) when live.py isn't running or there's nothing new.
"""
import os, sys

LIVE_FILE  = os.path.expanduser("~/transcripts/live.txt")
STATE_FILE = os.path.expanduser("~/transcripts/.inject_cursor")
MAX_CHARS  = 4000

if not os.path.exists(LIVE_FILE):
    sys.exit(0)

file_size = os.path.getsize(LIVE_FILE)
if file_size == 0:
    sys.exit(0)

cursor = 0
if os.path.exists(STATE_FILE):
    try:
        cursor = int(open(STATE_FILE).read().strip())
    except (ValueError, FileNotFoundError):
        cursor = 0

if cursor > file_size:
    cursor = 0

with open(LIVE_FILE) as f:
    f.seek(cursor)
    new_text = f.read()
    new_cursor = f.tell()

if not new_text.strip():
    sys.exit(0)

if len(new_text) > MAX_CHARS:
    lines = new_text.strip().split("\n")
    kept = []
    total = 0
    for line in reversed(lines):
        if total + len(line) + 1 > MAX_CHARS:
            break
        kept.append(line)
        total += len(line) + 1
    kept.reverse()
    new_text = "[...earlier lines truncated...]\n" + "\n".join(kept)
print(f"<live-transcript>\n{new_text.strip()}\n</live-transcript>")

with open(STATE_FILE, "w") as f:
    f.write(str(new_cursor))
