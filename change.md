# Cheat Mind — this update adds

1. **Personalization** — sidebar: companion name, tone (Balanced/Playful/Calm & gentle),
   applied to every mode via `apply_personalization()`.
2. **Reply language** — sidebar dropdown: Auto (match user) / English / Wolof / Mandinka /
   French. Applies to chat replies and the comfort story.
3. **Mood trends chart** — an expander above the chat ("📊 Mood trends this session") shows
   a bar chart of `user_mood` readings once there are 2+ turns. Descriptive only, not
   diagnostic — caption says so explicitly.
4. **RAG-grounded Study Help subjects** — Study Help now grounds answers in real reference
   material for 11 subjects, not just Civic Education: Mathematics (Core), Physics,
   Chemistry, Biology, Literature in English, History, Government, Financial Accounting,
   Commerce, and Economics. Each has its own small local knowledge base
   (`<subject>_kb.jsonl`), embedded into a local Chroma store per subject via
   `get_subject_collection()`. When you type a question in a subject that has a KB file,
   the most relevant passages are retrieved and folded into the model's answer. English
   Language has no KB — grammar/comprehension/essay help relies on the model's general
   skill, which fits that subject better than fact lookup.
5. **Study timer** — inside Study Help mode, an "⏱️ Study timer & session log" expander
   lets you set minutes, start a countdown (runs client-side in the browser, chimes when
   time's up), and stop it early if needed.
6. **Google Sheets study log** — optionally log completed study sessions (timestamp,
   subject, minutes studied, note, mood) to a Google Sheet.

## New dependencies

Add to your requirements.txt:

```
chromadb
gspread
google-auth
```

`gspread`/`google-auth` are optional — if not installed, the sidebar shows a note and
Google Sheets logging is disabled, but everything else in the app still works fine.

## Setting up Google Sheets logging

1. In Google Cloud Console, create a service account and download its JSON key.
2. Enable the Google Sheets API for that project.
3. Open the target Google Sheet, click Share, and add the service account's email
   (found inside the JSON file, looks like `xxx@xxx.iam.gserviceaccount.com`) as an Editor.
4. In the app's sidebar, upload the JSON key and paste the sheet's URL.
5. The app auto-creates a "Cheat Mind Study Log" worksheet inside that sheet on first log.


## Important: keep all knowledge-base files next to app.py

All `*_kb.jsonl` files (civic_education, math, physics, chemistry, biology, literature,
history, government, accounting, commerce, economics) must sit in the same folder as
`cheat_mind_app.py` — the app reads each one relative to its own file location. All are
included here.

## Note on the knowledge bases

These are compact starting points (5-8 entries each covering core formulas/facts), not
complete or verified exam references — good enough to demo grounding and cover common
questions, but worth expanding and double-checking against the actual WASSCE syllabus
before treating answers as authoritative for real exam prep.
