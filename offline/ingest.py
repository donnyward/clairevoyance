#!/usr/bin/env python3
"""Copy m4a files from /tmp into date-organized folders and extract Voice Memo transcripts."""

import glob
import json
import os
import re
import shutil
import struct
import sys
from datetime import datetime

DATE_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})_")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}")


def skip(msg):
    print(f"[{datetime.now():%H:%M:%S}] SKIP {msg}")


def warn(msg):
    print(f"[{datetime.now():%H:%M:%S}] WARN {msg}", file=sys.stderr)


def err(msg):
    print(f"[{datetime.now():%H:%M:%S}] ERROR {msg}", file=sys.stderr)


# ----------
# Phase 1
# ----------
def move_from_tmp(root_dir):
    log("=== Phase 1: Move m4a files from /tmp ===")
    for path in sorted(glob.glob("/tmp/*.m4a")):
        fname = os.path.basename(path)
        m = DATE_RE.match(fname)
        if not m:
            warn(f"Cannot parse date from filename: {fname} -- skipping")
            continue

        date_dir = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        target_dir = os.path.join(root_dir, date_dir)
        target_file = os.path.join(target_dir, fname)

        if os.path.isfile(target_file):
            skip(f"{fname} (already in {date_dir}/)")
            continue

        os.makedirs(target_dir, exist_ok=True)
        shutil.move(path, target_file)
        log(f"MOVED {fname} -> {date_dir}/")


# ----------
# Phase 2
# ----------
def find_atom(f, path_parts, depth=0, end_pos=None):
    if depth >= len(path_parts):
        return True
    target = path_parts[depth].encode("ascii")
    start = f.tell()
    if end_pos is None:
        f.seek(0, 2)
        end_pos = f.tell()
        f.seek(start)

    while f.tell() < end_pos:
        pos = f.tell()
        header = f.read(8)
        if len(header) < 8:
            return None
        size_raw, atom_type = struct.unpack(">I4s", header)
        if size_raw == 0:
            atom_end = end_pos
        elif size_raw == 1:
            ext = f.read(8)
            if len(ext) < 8:
                return None
            atom_end = pos + struct.unpack(">Q", ext)[0]
        else:
            atom_end = pos + size_raw

        if atom_type == target:
            if depth == len(path_parts) - 1:
                data_start = f.tell()
                return (data_start, atom_end - data_start)
            result = find_atom(f, path_parts, depth + 1, atom_end)
            if result:
                return result

        f.seek(atom_end)
    return None


def extract_voicememo(m4a_path, output_path):
    with open(m4a_path, "rb") as f:
        result = find_atom(f, ["moov", "trak", "udta", "tsrp"])
        if not result:
            return "NO_TSRP"
        offset, size = result
        f.seek(offset)
        raw = f.read(size)

        json_start = raw.find(b"{")
        if json_start < 0:
            return "NO_JSON"

        data = json.loads(raw[json_start:].decode("utf-8", errors="replace"))
        astr = data.get("attributedString", {})
        attr_table = astr.get("attributeTable", [])
        runs = astr.get("runs", [])

        if not runs or not attr_table:
            return "EMPTY_RUNS"

        parts = []
        prev_end = None
        for i in range(0, len(runs), 2):
            text = runs[i]
            idx = runs[i + 1]
            entry = attr_table[idx]
            t_start = entry["timeRange"][0]
            t_end = entry["timeRange"][1]
            if prev_end is not None and (t_start - prev_end) > 2.0:
                parts.append("\n\n")
            parts.append(text)
            prev_end = t_end

        with open(output_path, "w") as out:
            out.write("".join(parts).strip() + "\n")

        return "OK"


def extract_all_voicememos(root_dir):
    log("=== Phase 2: Extract Voice Memo transcripts ===")
    m4a_files = []
    for dirpath, _dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith(".m4a"):
                m4a_files.append(os.path.join(dirpath, f))
    m4a_files.sort()

    for m4a_path in m4a_files:
        stem = m4a_path[: -len(".m4a")]
        voicememo_txt = f"{stem}_voicememo.txt"
        fname = os.path.basename(m4a_path)

        if os.path.isfile(voicememo_txt):
            skip(os.path.basename(voicememo_txt))
            continue

        try:
            result = extract_voicememo(m4a_path, voicememo_txt)
        except Exception as e:
            err(f"{fname}: tsrp extraction failed: {e}")
            continue

        if result == "OK":
            log(f"EXTRACTED voicememo from {fname}")
        elif result == "NO_TSRP":
            warn(f"{fname}: no tsrp atom found (no Voice Memo transcript generated?)")
        elif result == "NO_JSON":
            warn(f"{fname}: tsrp atom found but contains no JSON data")
        elif result == "EMPTY_RUNS":
            warn(f"{fname}: tsrp atom found but transcript is empty")


# ----------
# Main
# ----------
def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <root_dir>", file=sys.stderr)
        sys.exit(1)

    root_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    move_from_tmp(root_dir)
    extract_all_voicememos(root_dir)
    log("=== Done ===")


if __name__ == "__main__":
    main()
