# Speaker Identification Research

Research notes for adding persistent speaker identification to the transcription pipeline, so speakers get named automatically across recordings instead of being prompted for every time.

## Current flow (recap)

1. `transcribe.sh` runs WhisperX (with `--diarize`) on every audio file that lacks a `_whisper.json`.
2. `generate_transcripts.py` reads each JSON, finds the longest contiguous run of words per `SPEAKER_XX` label, prompts the user to name each one, and writes `_whisper.txt`.

WhisperX uses pyannote under the hood for diarization. That matters: pyannote's own embedding model is drop-in compatible with whatever WhisperX already clustered, so we don't need to pick a fresh ecosystem.

## The pattern has a name: speaker enrollment + identification

- **Enrollment**: first time you see a speaker, extract a voice embedding from a clean utterance and store it under a label (the person's name).
- **Identification**: for a new utterance, extract its embedding, compare against all enrolled embeddings via a similarity metric, accept the best match if above a threshold.

Everything below is variations on those two steps.

## Math primer (what the rest assumes you understand)

- **Embedding**: a fixed-length list of numbers produced by a neural net. A voice embedding is typically 192–512 numbers. Think of it as coordinates in that many dimensions.
- **L2 norm**: the length of a vector, generalized to any dimensions. `sqrt(sum of squares of each element)`.
- **L2 normalize**: divide each element by the L2 norm so the vector has length 1. Preserves direction, strips magnitude. Magnitude in voice embeddings is mostly noise (loudness, mic gain); direction encodes identity.
- **Cosine similarity**: cosine of the angle between two vectors. Range -1 to 1. 1 = identical direction, 0 = perpendicular (unrelated), -1 = opposite. Formula: `(A · B) / (||A|| * ||B||)`.
- **Dot product**: `A · B = sum(A[i] * B[i])`. For L2-normalized vectors, this equals cosine similarity directly (the denominator becomes 1), which is why we normalize upfront — one cheap multiply-and-sum replaces division by two norms.
- **Higher dot product = more similar** (for L2-normalized vectors). Don't confuse with cosine *distance* (`1 - similarity`), where lower is more similar.

## Step 1: Pick an embedding model

Three reasonable choices:

| Model | Library | Quality | Effort | Notes |
|---|---|---|---|---|
| `pyannote/wespeaker-voxceleb-resnet34-LM` | `pyannote.audio` | Best | Low | What WhisperX's diarizer uses internally. Same HF token. 256-dim. |
| `speechbrain/spkrec-ecapa-voxceleb` | `speechbrain` | Very good | Low | ECAPA-TDNN, no HF token needed. 192-dim. |
| Resemblyzer | `resemblyzer` | OK | Lowest | GE2E (2018), older but dead simple API. 256-dim. |

**Recommendation**: start with pyannote's wespeaker model. Using the same model WhisperX's diarizer used means "speakers WhisperX grouped together" and "speakers we're matching across files" live in the same embedding space — free consistency. SpeechBrain is the fallback if you ever want to drop the HF token dependency.

All three produce L2-normalized vectors out of the box, so cosine similarity = dot product. Don't overthink the metric.

## Step 2: What audio to feed the embedding model

You already have word-level timestamps per speaker. For a good embedding:

- **≥ 5 seconds** of contiguous speech from that speaker. 10–20s is safer.
- **No overlap / crosstalk** if possible. A segment with two voices produces a blended embedding that matches nobody.
- **16 kHz mono** — the format `transcribe.sh` already produces at `/tmp/whisper_convert_*.wav`.

`generate_transcripts.py`'s current `find_longest_line` finds the longest contiguous word run but only tracks `start`, not `end`. Extending it to return `(start, end)` is small. Audio slicing: ffmpeg (`-ss start -to end`) or `soundfile.read()` plus array slicing.

**Better than "one longest segment"**: grab the top-3 longest contiguous runs for each speaker and mean-pool their embeddings. Damps out flukes (a coughed segment, a laugh, a loud keyboard).

## Step 3: The "database"

At your scale — hundreds of recordings, ~tens of unique people — don't reach for a vector DB. Options in order of simplicity:

1. **Single JSON file** (+ `.npy` siblings for numpy arrays, or base64 in the JSON). Load once, linear scan, done. Fastest to build.
2. **SQLite** with embeddings as BLOB columns. Slightly more structure; nice if you want per-embedding metadata (source file, timestamp, score at enrollment, auto vs. confirmed).
3. **FAISS / Chroma / LanceDB** — overkill below low-thousands of speakers. Skip.

Conceptual schema:

```
speakers:
  name: "Kevin Broas"
  embeddings: [
    { vec: [...], source: "20210113_.../file.m4a", start: 123.4, end: 145.2, auto: false },
    ...
  ]
  centroid: [... mean of vecs, L2-normalized ...]
```

Store multiple embeddings per person, not one. Voices drift across recordings (mic, mood, mic gain, phone vs. room). Match against the **centroid** by default; keep individuals so you can recompute centroids when new samples arrive or audit mistakes.

## Step 4: Matching policy — two thresholds, not one

Single-threshold matching is a trap: you either auto-accept wrong matches (low threshold) or prompt for everything (high threshold). Use:

- **T_high** (auto-accept): above this, use the name without asking. For ECAPA / wespeaker / cosine, start around **0.80**.
- **T_low** (auto-reject): below this, treat as new speaker. Start around **0.55**.
- **Between**: prompt, but show the top-3 candidates and their scores. User accepts one or types a new name.

These numbers need calibration on your own data. Published thresholds (VoxCeleb EER etc.) are measured on short clean utterances; your segments are longer and meeting-mic-noisy. Empirical tuning beats literature numbers.

**How to calibrate**: run in "suggest-only" mode for your first ~20 files. Log `(top_match_name, score, user_correction)`. Plot histograms of scores for correct-match vs. wrong-match cases; the right T_high is roughly where the false-accept rate crosses whatever you're comfortable with (say 1–5%).

## Step 5: Integration into `generate_transcripts.py`

Conceptually, one new stage slots in between "find longest line per speaker" and "prompt for name":

1. For each anonymous `SPEAKER_XX`, grab the top-N longest segments' audio.
2. Embed each segment, mean-pool, L2-normalize → one query vector.
3. Query the DB. Get top-3 matches with scores.
4. Branch on thresholds:
   - `score ≥ T_high` → auto-assign. Tell the user what happened ("SPEAKER_00 → Kevin Broas [auto, 0.87]"). Optional: batch confirmation at the end.
   - Between → prompt with candidates pre-filled ("Likely Kevin (0.72), Bruce (0.61). Enter name, confirm, or 'new':").
   - `score < T_low` → prompt as today; save the embedding under the entered name.
5. Either way, if the entered/confirmed name is known, append this new embedding to that speaker's record and recompute the centroid. If new name, create a new record.

Additive only — if embedding extraction fails, fall back to the existing prompt.

## Step 6: Failure modes and countermeasures

- **Short segments** → bad embeddings. Enforce a minimum duration (~3s hard, ~5s soft). If no segment meets it, fall back to the prompt.
- **Overlapping speakers that WhisperX mis-merged** → poisoned embedding. Optional extra: run pyannote's VAD on the chosen chunk and require >90% single-speaker.
- **Compounding errors**: if you auto-accept a wrong match and save its embedding, the centroid drifts and future matches get worse. Mitigations: (a) conservative T_high at first; (b) log every auto-accept with timestamp and score for audit; (c) flag speakers whose embeddings have unusually high internal variance.
- **Same-family / similar voices**: genuinely hard. Two-threshold "between" band naturally catches these.
- **Mic / codec drift across recordings**: loudness-normalize audio (`pyloudnorm`, or `ffmpeg -af loudnorm`) before embedding. Helps a surprising amount across meeting platforms.
- **Longest = loudest, not most representative**: averaging across multiple long segments (not just the single longest) helps.

## Step 7: Incremental rollout path

Don't build it all at once:

1. **v0 — passive enrollment**. Add embedding extraction and DB save, but still always prompt. Each prompt also saves the embedding under the entered name. Spend a week just building the DB. Zero risk of wrong auto-matches.
2. **v1 — suggest mode**. Show top-k candidates at the prompt, don't auto-accept. Manually tune thresholds while accumulating labeled pairs.
3. **v2 — auto-accept**. Turn on auto-accept above T_high. Keep logs of auto-decisions.
4. **v3 — re-enrollment pass**. Periodically recompute centroids across accumulated samples, drop outliers.

This also gives you free training data for picking `T_high` / `T_low`: after v1 you have `(match_score, was_correct)` pairs.

## Recommended starting stack

- **Embedding**: `pyannote.audio` + `pyannote/wespeaker-voxceleb-resnet34-LM` (same HF ecosystem you already have).
- **Audio**: `soundfile` for slicing the 16 kHz WAV that `transcribe.sh` already produces.
- **Storage**: one SQLite file (`speakers.db`) at the repo root, schema above.
- **Similarity**: cosine via numpy dot-product on L2-normalized vectors.
- **Policy**: two-threshold (start 0.80 / 0.55), prompt-with-candidates in the middle band.
- **Mode**: suggest-only for the first ~20 files, auto-accept after calibration.

Resist adding a vector DB, a neural reranker, or a fancy active-learning loop. At this scale the boring version — JSON/SQLite + cosine + two thresholds + multiple embeddings per speaker — solves 95% of the problem.

## Side note: HF token

`transcribe.sh` hardcodes the HuggingFace token. Fine for a local-only repo. Externalize it (env var or `.env`) and rotate the token before pushing anywhere public.
