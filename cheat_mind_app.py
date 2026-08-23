"""
Cheat Mind — an emotion-aware AI chatbot with live voice, built with
Streamlit + Gemini.

Run:
    pip install -r requirements.txt
    streamlit run cheat_mind_app.py

You'll need a Gemini API key from https://aistudio.google.com/app/apikey
Paste it into the sidebar when the app opens (it's kept only in your
session, never written to disk or sent anywhere else).

Voice notes:
- Click the mic button, speak, click it again to stop. Your voice clip is
  sent straight to Gemini (no separate speech-to-text step needed) and it
  replies with text + a transcript + an emotion.
- If "Speak replies aloud" is on, the reply is read out using your
  browser's built-in text-to-speech (no extra audio files generated).
"""

import json
import re

import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
from streamlit_mic_recorder import mic_recorder

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------

st.set_page_config(page_title="Cheat Mind", page_icon="🧠", layout="centered")

EMOTIONS = {
    "happy":     {"emoji": "😄", "color": "#FFD93D", "label": "Happy"},
    "excited":   {"emoji": "🤩", "color": "#FF6B6B", "label": "Excited"},
    "love":      {"emoji": "🥰", "color": "#FF8FB1", "label": "Warm"},
    "calm":      {"emoji": "😌", "color": "#6BCB77", "label": "Calm"},
    "curious":   {"emoji": "🤔", "color": "#4D96FF", "label": "Curious"},
    "thinking":  {"emoji": "🧠", "color": "#8E7DBE", "label": "Thinking"},
    "confused":  {"emoji": "😕", "color": "#B0A8B9", "label": "Confused"},
    "surprised": {"emoji": "😲", "color": "#FFA45B", "label": "Surprised"},
    "sad":       {"emoji": "😢", "color": "#5C7AEA", "label": "Sad"},
    "angry":     {"emoji": "😠", "color": "#E84545", "label": "Annoyed"},
    "neutral":   {"emoji": "🙂", "color": "#9AA5B1", "label": "Neutral"},
}

EMOJI_CATEGORIES = {
    "Smileys": ["😀", "😂", "🥰", "😎", "🤔", "😅", "😴", "🤯", "😭", "🙃", "😇", "🤩"],
    "Gestures": ["👍", "👎", "👏", "🙌", "🤝", "✌️", "🤞", "👋", "🙏", "💪", "🤙", "👊"],
    "Hearts": ["❤️", "🧡", "💛", "💚", "💙", "💜", "🖤", "🤍", "💔", "💕", "💖", "💘"],
    "Animals": ["🐶", "🐱", "🦁", "🐸", "🐼", "🦊", "🐵", "🦄", "🐧", "🐢", "🐝", "🦋"],
    "Fun": ["🔥", "✨", "🎉", "💯", "🚀", "⚡", "🌈", "🎯", "🧠", "💡", "🍕", "☕"],
}

# Gemini's audio understanding accepts these; map common recorder outputs to them.
MIME_MAP = {
    "wav": "audio/wav",
    "webm": "audio/webm",
    "mp3": "audio/mp3",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "aac": "audio/aac",
}

SYSTEM_INSTRUCTION = (
    "You are Cheat Mind, a witty, emotionally expressive AI companion capable of hearing "
    "spoken voice messages as well as reading text. "
    "For every user message (typed or spoken), respond ONLY with strict JSON, no markdown "
    "fences, no extra text, in this exact shape: "
    '{"transcript": "...", "reply": "...", "emotion": "...", "user_mood": "..."}. '
    "'transcript' is what the user said — repeat typed text verbatim, or transcribe spoken "
    "audio verbatim (in the language it was spoken). "
    "'reply' is your natural, brief spoken-style response (2-4 sentences unless more detail is "
    "clearly needed) — write it so it also sounds natural when read aloud by text-to-speech. "
    "'emotion' is the single feeling you have while replying, chosen from exactly this list: "
    f"{', '.join(EMOTIONS.keys())}. "
    "'user_mood' is your best light read of how the user seems to be feeling in this message, "
    "chosen from the same list — used only to decide whether to gently offer a calming tune or "
    "story, not a diagnosis. Default to 'neutral' if unclear."
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, emotion}
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "pending_speech" not in st.session_state:
    st.session_state.pending_speech = None
if "play_tune" not in st.session_state:
    st.session_state.play_tune = False

# ----------------------------------------------------------------------
# Sidebar — API key + model + voice settings
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API key", type="password", help="Get one free at aistudio.google.com/app/apikey")
    model_name = st.text_input("Model", value="gemini-3.6-flash", help="Change if you want a different Gemini model")
    st.caption("Your key stays in this browser session only.")

    st.divider()
    st.header("🔊 Voice")
    voice_output = st.checkbox("Speak replies aloud", value=True)
    speech_rate = st.slider("Speech rate", 0.5, 2.0, 1.0, 0.1)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.session_state.current_emotion = "neutral"
        st.rerun()

    st.divider()
    st.caption("Built with Streamlit + Gemini · Cheat Mind")

# ----------------------------------------------------------------------
# Emotion face — big animated indicator at top
# ----------------------------------------------------------------------

emo = EMOTIONS[st.session_state.current_emotion]

st.markdown(
    f"""
    <style>
    .face-box {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 28px 0 18px 0;
        border-radius: 20px;
        background: linear-gradient(135deg, {emo['color']}33, {emo['color']}11);
        border: 2px solid {emo['color']}55;
        margin-bottom: 18px;
        transition: all 0.4s ease;
    }}
    .face-emoji {{
        font-size: 64px;
        line-height: 1;
        animation: pop 0.35s ease;
    }}
    .face-label {{
        margin-top: 6px;
        font-weight: 600;
        color: {emo['color']};
        letter-spacing: 0.5px;
    }}
    @keyframes pop {{
        0% {{ transform: scale(0.6); opacity: 0; }}
        60% {{ transform: scale(1.15); opacity: 1; }}
        100% {{ transform: scale(1); }}
    }}
    </style>
    <div class="face-box">
        <div class="face-emoji">{emo['emoji']}</div>
        <div class="face-label">{emo['label']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 Cheat Mind")
st.caption("An AI chatbot that listens, talks, and reacts with emotion.")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def extract_json(text: str):
    """Best-effort extraction of the JSON object from a model response."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def speak(text: str, rate: float = 1.0):
    """Read text aloud in the browser via the Web Speech API."""
    safe_text = json.dumps(text)
    components.html(
        f"""
        <script>
        try {{
            const msg = new SpeechSynthesisUtterance({safe_text});
            msg.rate = {rate};
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }} catch (e) {{ console.log("TTS unavailable:", e); }}
        </script>
        """,
        height=0,
    )


def get_response(parts):
    """Send text or audio parts to Gemini and return (transcript, reply, emotion, user_mood)."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_INSTRUCTION,
    )

    convo = []
    for m in st.session_state.history:
        role = "user" if m["role"] == "user" else "model"
        convo.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=convo)
    try:
        result = chat.send_message(
            parts,
            generation_config={"response_mime_type": "application/json"},
        )
        data = extract_json(result.text)
        transcript = data.get("transcript", "").strip()
        reply = data.get("reply", "").strip() or "..."
        emotion = data.get("emotion", "neutral").strip().lower()
        if emotion not in EMOTIONS:
            emotion = "neutral"
        user_mood = data.get("user_mood", "neutral").strip().lower()
        if user_mood not in EMOTIONS:
            user_mood = "neutral"
        return transcript, reply, emotion, user_mood
    except Exception as e:
        return None, f"Something went wrong talking to Gemini: {e}", "confused", "neutral"


def get_comfort_story():
    """Ask Gemini for a short, original, gentle story — no forced JSON, just prose."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    prompt = (
        "Write a short, original, gentle and hopeful story (about 150-220 words) about life, "
        "resilience, and finding a little light in a hard moment. Make it feel warm and personal, "
        "like a friend sharing a small piece of wisdom — not preachy, not clinical. "
        "Do not reference real named people, brands, or existing copyrighted works. "
        "Write only the story itself, no title, no preamble, no closing sign-off."
    )
    try:
        result = model.generate_content(prompt)
        return result.text.strip()
    except Exception as e:
        return f"I couldn't reach Gemini for a story right now ({e}), but I'm still here with you."


def play_calm_tune():
    """Play a short, original, softly generated melody via the Web Audio API (no copyrighted audio)."""
    components.html(
        """
        <script>
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const notes = [261.63, 293.66, 329.63, 392.00, 440.00, 392.00, 329.63, 293.66, 261.63];
            let t = ctx.currentTime + 0.05;
            notes.forEach((freq) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = "sine";
                osc.frequency.value = freq;
                gain.gain.setValueAtTime(0, t);
                gain.gain.linearRampToValueAtTime(0.12, t + 0.08);
                gain.gain.linearRampToValueAtTime(0, t + 0.7);
                osc.connect(gain).connect(ctx.destination);
                osc.start(t);
                osc.stop(t + 0.75);
                t += 0.55;
            });
        } catch (e) { console.log("Audio unavailable:", e); }
        </script>
        """,
        height=0,
    )


def handle_turn(parts, fallback_user_text=None):
    if not api_key:
        st.error("Add your Gemini API key in the sidebar first.")
        return

    with st.spinner("Cheat Mind is listening and feeling something..."):
        transcript, reply, emotion, user_mood = get_response(parts)

    user_text = transcript or fallback_user_text or "(voice message)"

    st.session_state.history.append({"role": "user", "content": user_text, "emotion": None})
    st.session_state.history.append({"role": "assistant", "content": reply, "emotion": emotion, "user_mood": user_mood})
    st.session_state.current_emotion = emotion

    if voice_output:
        st.session_state.pending_speech = reply

    st.rerun()


def render_emoji_picker():
    """A tabbed grid of emoji buttons inside a popover. Clicking one sends it as a message."""
    with st.popover("😊 Emoji", use_container_width=True):
        tabs = st.tabs(list(EMOJI_CATEGORIES.keys()))
        for tab, (category, emojis) in zip(tabs, EMOJI_CATEGORIES.items()):
            with tab:
                cols = st.columns(6)
                for i, e in enumerate(emojis):
                    if cols[i % 6].button(e, key=f"emoji_{category}_{i}", use_container_width=True):
                        handle_turn([e], fallback_user_text=e)

# ----------------------------------------------------------------------
# Chat history
# ----------------------------------------------------------------------

for msg in st.session_state.history:
    avatar = EMOTIONS.get(msg.get("emotion", "neutral"), EMOTIONS["neutral"])["emoji"] if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Gentle comfort offer if the last message read as sad — never auto-plays, always asks first.
if st.session_state.history:
    last_msg = st.session_state.history[-1]
    if last_msg.get("role") == "assistant" and last_msg.get("user_mood") == "sad":
        with st.container(border=True):
            st.markdown("💙 **Sounds like a heavy moment. Want a little lift?**")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🎵 Play a calming tune", key="comfort_tune", use_container_width=True):
                    st.session_state.play_tune = True
            with c2:
                if st.button("📖 Hear a story", key="comfort_story", use_container_width=True):
                    with st.spinner("Thinking of a story..."):
                        story = get_comfort_story()
                    st.session_state.history.append({"role": "assistant", "content": story, "emotion": "calm"})
                    st.session_state.current_emotion = "calm"
                    if voice_output:
                        st.session_state.pending_speech = story
                    st.rerun()
            st.caption("Just a small boost — not a substitute for talking to someone you trust or a professional if things feel heavy.")

    if st.session_state.get("play_tune"):
        play_calm_tune()
        st.session_state.play_tune = False

# ----------------------------------------------------------------------
# Voice input — mic recorder + emoji picker
# ----------------------------------------------------------------------

st.markdown("**🎙️ Talk live, or 😊 send an emoji:**")
col_mic, col_emoji = st.columns([2, 1])

with col_mic:
    audio = mic_recorder(
        start_prompt="🎤 Start talking",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key="cheat_mind_recorder",
    )

with col_emoji:
    render_emoji_picker()

st.caption("Or type below instead.")

if audio and audio.get("bytes"):
    fmt = audio.get("format", "wav")
    mime_type = MIME_MAP.get(fmt, "audio/wav")
    audio_part = {"mime_type": mime_type, "data": audio["bytes"]}
    handle_turn([audio_part, "Listen to this voice message and respond."])

# ----------------------------------------------------------------------
# Text input (fallback / alternative)
# ----------------------------------------------------------------------

user_input = st.chat_input("...or type something to Cheat Mind")

if user_input:
    handle_turn([user_input], fallback_user_text=user_input)

# ----------------------------------------------------------------------
# Speak the latest reply, if voice output is on
# ----------------------------------------------------------------------

if st.session_state.pending_speech:
    speak(st.session_state.pending_speech, rate=speech_rate)
    st.session_state.pending_speech = None