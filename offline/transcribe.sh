#!/usr/bin/env bash
set -uo pipefail

export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_TOKEN="$(cat "$SCRIPT_DIR/../hf_token")"

trap 'rm -f /tmp/whisper_convert_$$_*.wav 2>/dev/null' EXIT

# --- Phase 1: discover audio files ---
all_files=()
while IFS= read -r f; do
  all_files+=("$f")
done < <(find . -type f \( -iname '*.m4a' -o -iname '*.mp3' -o -iname '*.flac' \) | sort)

total_all=${#all_files[@]}
if [ "$total_all" -eq 0 ]; then
  echo "No audio files found."
  exit 0
fi

# --- Phase 2: prompt for each file not yet transcribed ---
queue_files=()
queue_speakers=()
skipped_existing=0
skipped_user=0

idx=0
for src in "${all_files[@]}"; do
  idx=$((idx+1))
  dir="$(dirname "$src")"
  base="$(basename "$src" | sed 's/\.[^.]*$//')"

  if [ -f "$dir/${base}_whisper.json" ]; then
    echo "($idx/$total_all) already transcribed, skipping: $src"
    skipped_existing=$((skipped_existing+1))
    continue
  fi

  while :; do
    echo "" > /dev/tty
    printf "(%d/%d) %s\nHow many speakers? [Enter=auto-detect, number=fixed, 'skip'=skip file] " \
      "$idx" "$total_all" "$src" > /dev/tty
    read -r answer < /dev/tty
    if [[ -z "$answer" ]]; then
      queue_files+=("$src")
      queue_speakers+=("")
      break
    elif [[ "$answer" == "skip" ]]; then
      echo "  -> will skip this file" > /dev/tty
      skipped_user=$((skipped_user+1))
      break
    elif [[ "$answer" =~ ^[1-9][0-9]*$ ]]; then
      queue_files+=("$src")
      queue_speakers+=("$answer")
      break
    else
      echo "  invalid: enter a positive integer, press Enter, or type 'skip'" > /dev/tty
    fi
  done
done

# --- Phase 3: pre-flight summary + confirm ---
queued=${#queue_files[@]}
echo ""
echo "Pre-flight summary:"
echo "  Total audio files found:  $total_all"
echo "  Already transcribed:      $skipped_existing"
echo "  Marked skip by user:      $skipped_user"
echo "  Queued for processing:    $queued"
echo ""
if [ "$queued" -eq 0 ]; then
  echo "Nothing to do."
  exit 0
fi
printf "Proceed? [y/N] " > /dev/tty
read -r confirm < /dev/tty
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

# --- Phase 4: batch process ---
failed_files=()
started_at=$(date '+%Y-%m-%d %H:%M:%S')
echo "Batch started: $started_at"

for i in "${!queue_files[@]}"; do
  src="${queue_files[$i]}"
  spk="${queue_speakers[$i]}"
  n=$((i+1))
  dir="$(dirname "$src")"
  base="$(basename "$src" | sed 's/\.[^.]*$//')"
  tmp_wav="/tmp/whisper_convert_$$_${i}.wav"

  echo ""
  echo "=== ($n/$queued) Processing: $src === $(date '+%H:%M:%S')"
  if [[ -z "$spk" ]]; then
    echo "    speakers: auto-detect"
  else
    echo "    speakers: $spk (fixed)"
  fi

  ffmpeg_ok=false
  for attempt in 1 2 3; do
    if ffmpeg -y -i "$src" -ar 16000 -ac 1 -c:a pcm_s16le "$tmp_wav" </dev/null; then
      ffmpeg_ok=true
      break
    fi
    echo "ffmpeg failed (attempt $attempt)"
    if [ "$attempt" -lt 3 ]; then sleep 2; fi
  done
  if [ "$ffmpeg_ok" != true ]; then
    echo "!! ffmpeg gave up on: $src -- continuing to next file"
    failed_files+=("ffmpeg: $src")
    rm -f "$tmp_wav"
    continue
  fi

  wx_args=(
    --model large-v3
    --device cpu
    --batch_size 4
    --compute_type int8
    --diarize
    --hf_token "$HF_TOKEN"
    --align_model WAV2VEC2_ASR_LARGE_LV60K_960H
    --output_format json
    --language en
    --output_dir "$dir"
  )
  if [[ -n "$spk" ]]; then
    wx_args+=(--min_speakers "$spk" --max_speakers "$spk")
  fi

  echo "  running: uv run --project \"$SCRIPT_DIR\" whisperx ${wx_args[*]} \"$tmp_wav\""

  if ! uv run --project "$SCRIPT_DIR" whisperx "${wx_args[@]}" "$tmp_wav" </dev/null; then
    echo "!! whisperx failed on: $src -- continuing"
    failed_files+=("whisperx: $src")
    rm -f "$tmp_wav"
    continue
  fi

  produced="$dir/whisper_convert_$$_${i}.json"
  if [ -f "$produced" ]; then
    mv "$produced" "$dir/${base}_whisper.json"
    echo "=== ($n/$queued) Done: $dir/${base}_whisper.json ==="
  else
    echo "!! expected output not found: $produced"
    failed_files+=("no-output: $src")
  fi

  rm -f "$tmp_wav"
done

# --- Final report ---
echo ""
echo "=========================================="
echo "Batch started:  $started_at"
echo "Batch ended:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "Succeeded:      $(( queued - ${#failed_files[@]} ))/$queued"
echo "Failed:         ${#failed_files[@]}"
if [ "${#failed_files[@]}" -gt 0 ]; then
  echo "Failures:"
  for f in "${failed_files[@]}"; do
    echo "  - $f"
  done
fi
