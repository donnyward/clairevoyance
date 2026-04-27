#!/usr/bin/env python3
"""Rename files within each recording folder so the basename matches the folder name.

Preserves known suffixes (_whisper, _googlerecorder, _voicememo) and file extensions.
Errors on collisions.

Usage:
  python3 rename_dry_run.py            # dry run
  python3 rename_dry_run.py --execute  # actually rename
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path("/Users/donny/Documents/tonas/Recordings")
KNOWN_SUFFIXES = ["_whisper", "_googlerecorder", "_voicememo"]
FOLDER_PATTERN = re.compile(r"^\d{8}_\d{4,6}")


def split_name(filename: str):
    """Return (core, suffix, ext). suffix is '' if no known suffix matches."""
    stem, ext = os.path.splitext(filename)
    for suf in KNOWN_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)], suf, ext
    return stem, "", ext


def plan_folder(folder: Path):
    """Return (renames, error). renames is list of (old, new); error is str or None."""
    files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not files:
        return [], None

    target = folder.name
    renames = []
    seen_targets: dict[str, str] = {}

    for f in sorted(files):
        _, suffix, ext = split_name(f.name)
        new_name = f"{target}{suffix}{ext}"
        if new_name in seen_targets:
            return [], (
                f"collision: '{seen_targets[new_name]}' and '{f.name}' "
                f"would both become '{new_name}'"
            )
        seen_targets[new_name] = f.name
        if f.name != new_name:
            renames.append((f.name, new_name))

    return renames, None


def main():
    execute = "--execute" in sys.argv

    folders = sorted(
        d for d in ROOT.iterdir()
        if d.is_dir() and FOLDER_PATTERN.match(d.name)
    )

    total_renames = 0
    error_folders = []
    nochange_folders = 0

    for folder in folders:
        renames, error = plan_folder(folder)
        if error:
            error_folders.append((folder.name, error))
            print(f"\n[ERROR] {folder.name}/")
            print(f"  {error}")
            continue
        if not renames:
            nochange_folders += 1
            continue

        print(f"\n{folder.name}/")
        for old, new in renames:
            print(f"  {old}")
            print(f"    -> {new}")
            if execute:
                (folder / old).rename(folder / new)
        total_renames += len(renames)

    print("\n" + "=" * 60)
    print(f"folders scanned:    {len(folders)}")
    print(f"folders unchanged:  {nochange_folders}")
    print(f"folders w/ errors:  {len(error_folders)}")
    print(f"total renames:      {total_renames}")
    print(f"mode:               {'EXECUTE' if execute else 'DRY RUN'}")
    if error_folders:
        print("\nErrored folders:")
        for name, err in error_folders:
            print(f"  - {name}: {err}")


if __name__ == "__main__":
    main()
