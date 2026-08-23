# 🧠 Cheat Mind

An emotion-aware AI chatbot with live voice, built with **Streamlit** + **Gemini**.

Talk to it (or type), and it replies with a natural response, speaks it back aloud, and shows an animated emoji face that reacts with the emotion it "felt" while replying.

---

## Features

- 💬 **Text chat** — normal chat input, full conversation history.
- 🎤 **Live voice input** — record a voice clip with one click; it's sent straight to Gemini, which transcribes and understands it directly (no separate speech-to-text step).
- 🔊 **Spoken replies** — toggle in the sidebar to have replies read aloud using your browser's built-in text-to-speech.
- 🎭 **Animated emotion face** — a big emoji + color-coded panel at the top that changes based on the emotion Gemini reports with each reply (happy, curious, thinking, annoyed, etc.).
- 😊 **Emoji picker** — a quick popover with categorized emoji (Smileys, Gestures, Hearts, Animals, Fun). Tap one to send it straight into the chat as a message.
- 🔐 **No stored keys** — your Gemini API key is entered per-session in the sidebar and never written to disk.

---

## Setup

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get a Gemini API key**

   Free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

3. **Run the app**

   ```bash
   streamlit run cheat_mind_app.py
   ```

4. Paste your API key into the sidebar when the app opens in your browser.

---

## Usage

- **Type**: use the chat box at the bottom.
- **Talk**: click **🎤 Start talking**, speak, click **⏹ Stop**. Your browser will ask for microphone permission the first time.
- **Send an emoji**: click **😊 Emoji**, pick a category tab, tap an emoji — it's sent immediately as a message and Cheat Mind reacts to it.
- **Hear replies**: leave **"Speak replies aloud"** checked in the sidebar (on by default). Adjust the speech rate slider to taste.
- **Change model**: the sidebar lets you swap the Gemini model name if you want to try a different one.
- **Clear chat**: sidebar button resets the conversation and emotion state.

---

## Requirements

- Python 3.9+
- A Gemini API key
- Microphone access in your browser (for voice input)
- **HTTPS or `localhost`** — browsers block microphone access on plain HTTP over a network IP, so voice input only works when run locally or deployed behind HTTPS.

---

## Troubleshooting

**Mic button doesn't do anything / no permission prompt**
Make sure you're on `localhost` or an HTTPS deployment, and that your browser has mic access enabled for the site.

**"Something went wrong talking to Gemini" after a voice message**
This is usually an audio-format mismatch. Run:

```bash
pip show streamlit-mic-recorder
```

and check the version — older versions default to `wav`, newer ones may default to `webm`. The app auto-detects and maps this, but if it still fails, try pinning:

```bash
pip install streamlit-mic-recorder==0.0.8
```

**No sound when a reply comes back**
Browser text-to-speech (`speechSynthesis`) sometimes needs a page interaction first — click anywhere on the page once, then try again. Voice quality/availability also depends on your OS (Windows, macOS, Android, etc. all ship different default voices).

**API key errors**
Double check the key was copied fully and has no extra whitespace. Keys are session-only — refreshing the page clears it.

---

## Project structure

```
cheat_mind_app.py   # main Streamlit app
requirements.txt    # Python dependencies
README.md           # this file
```

---

## Roadmap ideas

- Mobile/vertical layout for demo recordings
- "Listening..." animation while recording
- Persistent chat history across sessions
- Custom TTS voice selection
