"""
Cheat Mind — an emotion-aware AI chatbot built with Streamlit + Gemini.

Run:
    pip install -r requirements.txt
    streamlit run cheat_mind_app.py

You'll need a Gemini API key from https://aistudio.google.com/app/apikey
Paste it into the sidebar when the app opens (it's kept only in your
session, never written to disk or sent anywhere else).
"""

import json
import re

import streamlit as st
import google.generativeai as genai

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

SYSTEM_INSTRUCTION = (
    "You are Cheat Mind, a witty, emotionally expressive AI companion. "
    "For every user message, reply naturally and briefly (2-5 sentences unless more detail is clearly needed), "
    "and pick the single emotion you feel while replying from this exact list: "
    f"{', '.join(EMOTIONS.keys())}. "
    "Respond ONLY with strict JSON, no markdown fences, no extra text, in this exact shape: "
    '{"reply": "your message here", "emotion": "one_of_the_list"}'
)

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content, emotion}
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"

# ----------------------------------------------------------------------
# Sidebar — API key + model settings
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API key", type="password", help="Get one free at aistudio.google.com/app/apikey")
    model_name = st.text_input("Model", value="gemini-2.0-flash", help="Change if you want a different Gemini model")
    st.caption("Your key stays in this browser session only.")

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
st.caption("An AI chatbot that reacts with emotion to every message.")

# ----------------------------------------------------------------------
# Chat history
# ----------------------------------------------------------------------

for msg in st.session_state.history:
    avatar = EMOTIONS.get(msg.get("emotion", "neutral"), EMOTIONS["neutral"])["emoji"] if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

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


def get_response(user_text: str):
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
            user_text,
            generation_config={"response_mime_type": "application/json"},
        )
        data = extract_json(result.text)
        reply = data.get("reply", "").strip() or "..."
        emotion = data.get("emotion", "neutral").strip().lower()
        if emotion not in EMOTIONS:
            emotion = "neutral"
        return reply, emotion
    except Exception as e:
        return f"Something went wrong talking to Gemini: {e}", "confused"

# ----------------------------------------------------------------------
# Chat input
# ----------------------------------------------------------------------

user_input = st.chat_input("Say something to Cheat Mind...")

if user_input:
    if not api_key:
        st.error("Add your Gemini API key in the sidebar first.")
    else:
        st.session_state.history.append({"role": "user", "content": user_input, "emotion": None})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(user_input)

        with st.spinner("Cheat Mind is feeling something..."):
            reply, emotion = get_response(user_input)

        st.session_state.current_emotion = emotion
        st.session_state.history.append({"role": "assistant", "content": reply, "emotion": emotion})

        with st.chat_message("assistant", avatar=EMOTIONS[emotion]["emoji"]):
            st.markdown(reply)

        st.rerun()
