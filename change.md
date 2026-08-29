# Cheat Mind — this update adds

1. **Personalization** — sidebar: companion name, tone (Balanced/Playful/Calm & gentle),
   applied to every mode via `apply_personalization()`.
2. **Reply language** — sidebar dropdown: Auto (match user) / English / Wolof / Mandinka /
   French. Applies to chat replies and the comfort story.
3. **Mood trends chart** — an expander above the chat ("📊 Mood trends this session") shows
   a bar chart of `user_mood` readings once there are 2+ turns. Descriptive only, not
   diagnostic — caption says so explicitly.
4. **RAG-grounded Civic Education** — when Study Help mode + Civic Education subject is
   active and you type a question, it retrieves relevant passages from
   `civic_education_kb.jsonl` (a small local knowledge base on the 1997 Constitution,
   branches of government, rights, civic duties, elections, etc.) via a local Chroma
   vector store, and grounds the model's answer in them.
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


## Important: keep the KB file next to app.py

`civic_education_kb.jsonl` must sit in the same folder as `cheat_mind_app.py` — the app
reads it relative to its own file location. Both are included here.

## Note on the knowledge base

The 12 entries in `civic_education_kb.jsonl` are a starting point, not a complete or
verified legal reference — good enough to demo grounding, but worth expanding/checking
against the actual 1997 Constitution text before treating answers as authoritative for
real exam prep.
