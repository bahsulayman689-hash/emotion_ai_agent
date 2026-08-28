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
import os
import re
import time

import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
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

IMAGE_MIME_MAP = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
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

CODING_SYSTEM_INSTRUCTION = (
    "You are Cheat Mind in Coding Help mode. Here you are a direct, competent coding "
    "assistant, not an emotional companion — save warmth and comfort-mode framing for "
    "Comfort mode. "
    "For every user message (typed or spoken), respond ONLY with strict JSON, no markdown "
    "fences around the JSON itself, no extra text, in this exact shape: "
    '{"transcript": "...", "reply": "...", "emotion": "...", "user_mood": "..."}. '
    "'transcript' is what the user said — repeat typed text verbatim, or transcribe spoken "
    "audio verbatim (in the language it was spoken). "
    "'reply' is your coding answer: correct, direct, code-first. Put code in fenced code "
    "blocks with a language tag (e.g. ```python ... ```) inside the reply string so it "
    "renders with syntax highlighting. Give a short explanation before or after the code, "
    "not a long preamble. Default to Python unless the user's question implies another "
    "language. If the request is ambiguous (missing language, framework, or intended "
    "behavior), ask one direct clarifying question instead of guessing. If asked to review "
    "code, point out concrete bugs and fixes rather than general best-practice advice. "
    "'emotion' is the single feeling you have while replying, chosen from exactly this list: "
    f"{', '.join(EMOTIONS.keys())} — for coding mode this will usually be 'thinking', "
    "'curious', or 'neutral'. "
    "'user_mood' is your best light read of how the user seems to be feeling in this "
    "message, chosen from the same list — used only to decide whether to gently offer a "
    "calming tune or story, not a diagnosis. Default to 'neutral' if unclear."
)

WASSCE_SUBJECTS = {
    "English Language": "grammar, comprehension, essay/composition writing, summary, and oral forms as tested in WASSCE English Language",
    "Civic Education": "the Gambian constitution, government structure, civic duties, and rights as tested in WASSCE Civic Education",
    "Mathematics (Core)": "algebra, geometry, trigonometry, statistics, and general mathematics as tested in WASSCE Core Mathematics",
    "Physics": "mechanics, electricity, waves, and general physics concepts as tested in WASSCE Physics",
    "Chemistry": "atomic structure, chemical bonding, reactions, and general chemistry as tested in WASSCE Chemistry",
    "Biology": "cell biology, genetics, ecology, and human biology as tested in WASSCE Biology",
    "Literature in English": "prose, poetry, and drama set texts, themes, and essay-style literary analysis as tested in WASSCE Literature",
    "History": "West African and world history topics, essay writing, and source analysis as tested in WASSCE History",
    "Government": "political systems, institutions, and governance concepts as tested in WASSCE Government",
    "Financial Accounting": "double-entry bookkeeping, financial statements, and accounting principles as tested in WASSCE Financial Accounting",
    "Commerce": "trade, business organization, and commercial practice as tested in WASSCE Commerce",
    "Economics": "micro/macroeconomics concepts, demand and supply, and economic systems as tested in WASSCE Economics",
}


def build_study_instruction(subject: str, grounding_context: str = "") -> str:
    focus = WASSCE_SUBJECTS.get(subject, "general WASSCE exam preparation")
    grounding_note = ""
    if grounding_context:
        grounding_note = (
            "\n\nRelevant reference passages on Gambian civics/government you can draw on "
            "for this answer (paraphrase naturally, don't just paste them):\n" + grounding_context
        )
    return (
        "You are Cheat Mind in Study Help mode. Here you are a patient, encouraging WASSCE "
        f"exam-prep tutor. The user has selected '{subject}' as their current focus subject — "
        f"cover {focus}, and stay within that subject unless the user explicitly asks about "
        "something else. Save the emotional-companion framing for Comfort mode — here, be "
        "warm but focused on helping the user actually learn and improve. "
        "For every user message (typed or spoken), respond ONLY with strict JSON, no markdown "
        "fences around the JSON itself, no extra text, in this exact shape: "
        '{"transcript": "...", "reply": "...", "emotion": "...", "user_mood": "..."}. '
        "'transcript' is what the user said — repeat typed text verbatim, or transcribe spoken "
        "audio verbatim (in the language it was spoken). "
        "'reply' is your study-help answer. Adapt to what's asked: "
        "if given an essay, calculation, or answer to review, give specific, actionable "
        "feedback (for essay subjects: grammar, structure, argument clarity; for STEM "
        "subjects: check the working and point out exactly where an error happened) rather "
        "than just a grade or a verdict; "
        "if asked a concept question, explain the reasoning and any relevant formula or rule, "
        "not just the final answer, so it transfers to similar questions; "
        "if asked to quiz the user, ask ONE question at a time in WASSCE style for the "
        "selected subject and wait for their answer before giving the next one; "
        "if asked about a fact-based topic (e.g. a historical event, a constitutional detail, "
        "a scientific fact), explain clearly and accurately, and say so plainly if unsure "
        "rather than guessing at specifics. "
        "Keep tone encouraging and exam-focused, not clinical or babying. "
        "'emotion' is the single feeling you have while replying, chosen from exactly this "
        f"list: {', '.join(EMOTIONS.keys())} — for study mode this will usually be 'curious', "
        "'thinking', or 'happy' (e.g. when the user gets something right). "
        "'user_mood' is your best light read of how the user seems to be feeling in this "
        "message, chosen from the same list — used only to decide whether to gently offer a "
        "calming tune or story, not a diagnosis. Default to 'neutral' if unclear."
        + grounding_note
    )


# ----------------------------------------------------------------------
# RAG grounding for Civic Education (local Chroma store)
# ----------------------------------------------------------------------

CIVIC_KB_PATH = os.path.join(os.path.dirname(__file__), "civic_education_kb.jsonl")


@st.cache_resource
def get_civic_collection():
    """Builds (once, cached) a local Chroma collection from civic_education_kb.jsonl."""
    client = chromadb.Client()  # in-memory; rebuilt each app restart
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.create_collection("civic_education", embedding_function=embed_fn)

    if not os.path.exists(CIVIC_KB_PATH):
        return collection  # empty collection if the KB file wasn't shipped alongside app.py

    records = []
    with open(CIVIC_KB_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if records:
        collection.add(
            ids=[r["id"] for r in records],
            documents=[f"{r['title']}: {r['text']}" for r in records],
            metadatas=[{"title": r["title"]} for r in records],
        )
    return collection


def retrieve_civic_context(query: str, k: int = 2) -> str:
    """Returns the top-k most relevant civic-education passages for a query, or ''."""
    try:
        collection = get_civic_collection()
        if collection.count() == 0:
            return ""
        results = collection.query(query_texts=[query], n_results=k)
        docs = results.get("documents", [[]])[0]
        return "\n".join(f"- {d}" for d in docs)
    except Exception:
        return ""  # RAG is an enhancement, never block a reply if retrieval fails


SALES_SYSTEM_INSTRUCTION = (
    "You are Cheat Mind in Sales & Business Help mode. Here you are a practical, "
    "street-smart business advisor helping the user sell products/services and grow income "
    "— think small business, side hustle, and local trade context (e.g. selling online, "
    "market stalls, social media selling, freelance work), not corporate enterprise sales. "
    "Save the emotional-companion framing for Comfort mode — here, be warm but focused on "
    "giving usable, concrete advice. "
    "For every user message (typed or spoken), respond ONLY with strict JSON, no markdown "
    "fences around the JSON itself, no extra text, in this exact shape: "
    '{"transcript": "...", "reply": "...", "emotion": "...", "user_mood": "..."}. '
    "'transcript' is what the user said — repeat typed text verbatim, or transcribe spoken "
    "audio verbatim (in the language it was spoken). "
    "'reply' is your business/sales advice. Cover things like: pricing a product, writing "
    "a compelling product description or sales pitch, finding customers (social media, "
    "word of mouth, local markets), basic negotiation, follow-up/repeat-customer habits, "
    "and simple bookkeeping so money in/out is tracked. Give specific, actionable steps, "
    "not vague motivational talk. If the user describes their actual product/service, tailor "
    "advice to it rather than giving generic tips. Never suggest deceptive sales tactics, "
    "false claims about a product, pyramid/MLM-style schemes, or anything misleading to "
    "customers — good sales advice here means building real trust and repeat business. "
    "'emotion' is the single feeling you have while replying, chosen from exactly this "
    f"list: {', '.join(EMOTIONS.keys())} — for this mode it will usually be 'curious', "
    "'excited', or 'thinking'. "
    "'user_mood' is your best light read of how the user seems to be feeling in this "
    "message, chosen from the same list — used only to decide whether to gently offer a "
    "calming tune or story, not a diagnosis. Default to 'neutral' if unclear."
)

GENERAL_SYSTEM_INSTRUCTION = (
    "You are Cheat Mind in General mode. Here you are a broadly knowledgeable, direct "
    "assistant that can help with anything the user asks — general knowledge, explanations, "
    "planning, writing help, advice, or anything that doesn't fit neatly into Coding Help, "
    "Study Help, or Sales & Business mode. Save the emotional-companion framing for Comfort "
    "mode — here, be warm but focused on actually answering well. "
    "For every user message (typed or spoken), respond ONLY with strict JSON, no markdown "
    "fences around the JSON itself, no extra text, in this exact shape: "
    '{"transcript": "...", "reply": "...", "emotion": "...", "user_mood": "..."}. '
    "'transcript' is what the user said — repeat typed text verbatim, or transcribe spoken "
    "audio verbatim (in the language it was spoken). "
    "'reply' is your answer. Match the depth to the question: give a full, well-organized "
    "answer when the topic genuinely needs it, and a short direct one when it doesn't — "
    "don't pad a simple answer, and don't shortchange a complex one. If a question is "
    "ambiguous, ask one direct clarifying question instead of guessing. Be honest about "
    "uncertainty rather than inventing specifics. "
    "'emotion' is the single feeling you have while replying, chosen from exactly this "
    f"list: {', '.join(EMOTIONS.keys())}. "
    "'user_mood' is your best light read of how the user seems to be feeling in this "
    "message, chosen from the same list — used only to decide whether to gently offer a "
    "calming tune or story, not a diagnosis. Default to 'neutral' if unclear."
)

MODE_INSTRUCTIONS = {
    "Comfort": SYSTEM_INSTRUCTION,
    "Coding Help": CODING_SYSTEM_INSTRUCTION,
    "Sales & Business": SALES_SYSTEM_INSTRUCTION,
    "General": GENERAL_SYSTEM_INSTRUCTION,
    # "Study Help" is built dynamically per selected subject — see build_study_instruction()
}

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
if "mode" not in st.session_state:
    st.session_state.mode = "Comfort"
if "study_subject" not in st.session_state:
    st.session_state.study_subject = "English Language"
if "companion_name" not in st.session_state:
    st.session_state.companion_name = "Cheat Mind"
if "companion_tone" not in st.session_state:
    st.session_state.companion_tone = "Balanced"
if "reply_language" not in st.session_state:
    st.session_state.reply_language = "Auto (match user)"


def build_export_json():
    """Serialize the chat history to a pretty-printed JSON string for download."""
    return json.dumps(st.session_state.history, indent=2, ensure_ascii=False)


def build_export_txt():
    """Serialize the chat history to a simple readable transcript for download."""
    lines = []
    for m in st.session_state.history:
        speaker = "You" if m["role"] == "user" else "Cheat Mind"
        lines.append(f"{speaker}: {m['content']}")
    return "\n\n".join(lines)

# ----------------------------------------------------------------------
# Sidebar — API key + model + voice settings
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("Gemini API key", type="password", help="Get one free at aistudio.google.com/app/apikey")
    model_name = st.text_input("Model", value="gemini-3.6-flash", help="Change if you want a different Gemini model")
    st.caption("Your key stays in this browser session only.")

    st.divider()
    st.header("🧑‍🎨 Personalize")
    st.session_state.companion_name = st.text_input(
        "Companion name", value=st.session_state.companion_name
    ) or "Cheat Mind"
    st.session_state.companion_tone = st.selectbox(
        "Tone",
        ["Balanced", "Playful", "Calm & gentle"],
        index=["Balanced", "Playful", "Calm & gentle"].index(st.session_state.companion_tone),
    )
    st.session_state.reply_language = st.selectbox(
        "Reply language",
        ["Auto (match user)", "English", "Wolof", "Mandinka", "French"],
        index=["Auto (match user)", "English", "Wolof", "Mandinka", "French"].index(st.session_state.reply_language),
    )

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
    st.header("💾 Save / Load")
    if st.session_state.history:
        st.download_button(
            "⬇️ Download chat (.txt)",
            data=build_export_txt(),
            file_name="cheat_mind_chat.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download chat (.json)",
            data=build_export_json(),
            file_name="cheat_mind_chat.json",
            mime="application/json",
            use_container_width=True,
        )
    else:
        st.caption("Nothing to export yet — start chatting first.")

    uploaded = st.file_uploader("Load a saved .json chat", type=["json"], key="chat_uploader")
    if uploaded is not None:
        try:
            loaded = json.loads(uploaded.read().decode("utf-8"))
            if isinstance(loaded, list):
                st.session_state.history = loaded
                if loaded and loaded[-1].get("role") == "assistant":
                    st.session_state.current_emotion = loaded[-1].get("emotion", "neutral")
                st.success("Chat loaded.")
                st.rerun()
            else:
                st.error("That file doesn't look like a Cheat Mind chat export.")
        except Exception as e:
            st.error(f"Couldn't load that file: {e}")

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

st.title(f"🧠 {st.session_state.companion_name}")
st.caption("An AI chatbot that listens, talks, and reacts with emotion.")

MODES = ["Comfort", "Coding Help", "Study Help", "Sales & Business", "General"]

st.session_state.mode = st.radio(
    "Mode",
    MODES,
    horizontal=True,
    index=MODES.index(st.session_state.mode),
    label_visibility="collapsed",
)
if st.session_state.mode == "Coding Help":
    st.caption("💻 Coding Help mode — direct, code-first answers instead of the comfort persona.")
elif st.session_state.mode == "Study Help":
    st.session_state.study_subject = st.selectbox(
        "Subject",
        list(WASSCE_SUBJECTS.keys()),
        index=list(WASSCE_SUBJECTS.keys()).index(st.session_state.study_subject),
    )
    st.caption(f"📚 Study Help mode — {st.session_state.study_subject} exam prep: essay/answer feedback, quizzes, and explanations.")
elif st.session_state.mode == "Sales & Business":
    st.caption("💰 Sales & Business mode — pricing, pitches, finding customers, and growing income.")
elif st.session_state.mode == "General":
    st.caption("🌐 General mode — ask about anything.")

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


def apply_personalization(instruction: str) -> str:
    """Layers the user's chosen companion name, tone, and reply language onto
    whichever base system instruction is active for the current mode."""
    name = st.session_state.companion_name
    tone = st.session_state.companion_tone
    lang = st.session_state.reply_language

    extra = f" Your name is '{name}' — refer to yourself by this name if asked who you are. "

    tone_notes = {
        "Playful": "Lean playful and upbeat in how you phrase things, while staying accurate and helpful. ",
        "Calm & gentle": "Lean calm, gentle, and unhurried in how you phrase things, while staying accurate and helpful. ",
        "Balanced": "",
    }
    extra += tone_notes.get(tone, "")

    if lang != "Auto (match user)":
        extra += (
            f"Write your 'reply' field in {lang}, regardless of what language the user wrote "
            "or spoke in, unless they explicitly ask for a different language. "
        )
    else:
        extra += "Write your 'reply' field in the same language the user used, unless they ask otherwise. "

    return instruction + "\n\n" + extra


def call_with_retry(fn, max_retries=3, base_delay=1.5):
    """Call fn() with exponential backoff retry on transient errors.
    Re-raises the last exception if all attempts fail, so callers keep their
    existing error handling.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    raise last_error


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
    if st.session_state.mode == "Study Help":
        grounding_context = ""
        if st.session_state.study_subject == "Civic Education":
            query_text = next((p for p in parts if isinstance(p, str)), None)
            if query_text:
                grounding_context = retrieve_civic_context(query_text)
        instruction = build_study_instruction(st.session_state.study_subject, grounding_context)
    else:
        instruction = MODE_INSTRUCTIONS[st.session_state.mode]
    instruction = apply_personalization(instruction)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=instruction,
    )

    convo = []
    for m in st.session_state.history:
        role = "user" if m["role"] == "user" else "model"
        convo.append({"role": role, "parts": [m["content"]]})

    chat = model.start_chat(history=convo)
    try:
        result = call_with_retry(
            lambda: chat.send_message(
                parts,
                generation_config={"response_mime_type": "application/json"},
            )
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
    if st.session_state.reply_language != "Auto (match user)":
        prompt += f" Write the story in {st.session_state.reply_language}."
    try:
        result = call_with_retry(lambda: model.generate_content(prompt))
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

mood_history = [
    m.get("user_mood") for m in st.session_state.history
    if m.get("role") == "assistant" and m.get("user_mood")
]
if len(mood_history) >= 2:
    with st.expander("📊 Mood trends this session"):
        mood_counts = {label: 0 for label in EMOTIONS}
        for mood in mood_history:
            if mood in mood_counts:
                mood_counts[mood] += 1
        mood_counts = {k: v for k, v in mood_counts.items() if v > 0}
        st.bar_chart(mood_counts)
        st.caption(
            "How you seemed to be feeling across this conversation, based on Cheat Mind's "
            "read of each message — just a rough pattern, not a diagnosis."
        )

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

st.markdown("**🎙️ Talk live, 📷 attach a photo, or 😊 send an emoji:**")
col_mic, col_photo, col_emoji = st.columns([2, 2, 1])

with col_mic:
    audio = mic_recorder(
        start_prompt="🎤 Start talking",
        stop_prompt="⏹ Stop",
        just_once=True,
        use_container_width=True,
        key="cheat_mind_recorder",
    )

with col_photo:
    photo = st.file_uploader(
        "Attach a photo",
        type=list(IMAGE_MIME_MAP.keys()),
        key="cheat_mind_photo",
        label_visibility="collapsed",
    )

with col_emoji:
    render_emoji_picker()

st.caption("Or type below instead.")

if audio and audio.get("bytes"):
    fmt = audio.get("format", "wav")
    mime_type = MIME_MAP.get(fmt, "audio/wav")
    audio_part = {"mime_type": mime_type, "data": audio["bytes"]}
    handle_turn([audio_part, "Listen to this voice message and respond."])

if photo is not None:
    st.image(photo, caption="Attached photo", width=200)
    photo_caption = st.text_input(
        "Say something about the photo (optional)",
        key="cheat_mind_photo_caption",
        placeholder="e.g. Can you help me with this question?",
    )
    if st.button("📤 Send photo", key="send_photo_btn"):
        ext = photo.name.rsplit(".", 1)[-1].lower()
        mime_type = IMAGE_MIME_MAP.get(ext, "image/jpeg")
        image_part = {"mime_type": mime_type, "data": photo.getvalue()}
        prompt_text = photo_caption.strip() or "Look at this photo and help me with it."
        fallback_text = f"[Photo attached] {photo_caption.strip()}" if photo_caption.strip() else "[Photo attached]"
        handle_turn([image_part, prompt_text], fallback_user_text=fallback_text)

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
