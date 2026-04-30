# Live Room Transcript

When `<live-transcript>` tags appear in your context, this is a real-time transcription of a conversation happening around the user. The user is one of the speakers. Use this context to:

1. Understand what's being discussed in the room
2. Answer questions that reference the conversation (e.g. "what did they just say about X?")
3. Proactively surface relevant information when it relates to your current work together

Format: `[HH:MM:SS] [Speaker N] text`. Speaker labels are consistent within a session but not across sessions. Only new lines since the last prompt are injected each time — prior transcript lines are in earlier conversation turns.

The transcription is imperfect — expect mistranscribed words, missing segments, and occasional hallucinated repetitions (e.g. repeated "Ta Ta Ta" or looping phrases). Use surrounding context to infer what was likely said rather than taking individual words literally. Some parts of the conversation may not appear at all if they were too quiet or overlapped.
