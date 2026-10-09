"""
Cheat Mind — an emotion-aware AI chatbot with live voice, built with
Streamlit + Gemini.

Run:
    pip install -r requirements.txt
    streamlit run cheat_mind_app.py

API keys (secure setup):
- On Streamlit Cloud: App -> Settings -> Secrets, add GEMINI_API_KEY
  (and optionally GSHEET_URL and GCP_SERVICE_ACCOUNT).
- Locally: put the same lines in .streamlit/secrets.toml and add that file
  to .gitignore. NEVER commit keys to GitHub.
- If no secret is set, the sidebar shows a box where a user can paste their
  own Gemini key (kept only in their session).
Get a key at https://aistudio.google.com/app/apikey

Voice notes:
- Click the mic button, speak, click it again to stop. Your voice clip is
  sent straight to Gemini (no separate speech-to-text step needed) and it
  replies with text + a transcript + an emotion.
- If "Speak replies aloud" is on, the reply is read out using your
  browser's built-in text-to-speech (no extra audio files generated).

Features:
- Quiz mode      - AI-generated WASSCE-style MCQs, scored, with explanations
- Progress mode  - accuracy per subject, score history, weak topics
- Study Plan     - exam countdown + AI day-by-day plan targeting weak topics
- Video Studio   - always uses the student's OWN paid key, never the shared one
"""

import json
import os
import re
import time
import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
from streamlit_mic_recorder import mic_recorder

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSHEETS_AVAILABLE = True
except ImportError:
    GSHEETS_AVAILABLE = False

try:
    from google import genai as veo_genai
    from google.genai import types as veo_types
    VEO_AVAILABLE = True
except ImportError:
    VEO_AVAILABLE = False

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

REPLY_LANGUAGES = [
    "Auto (match user)",
    "English",
    "Wolof",
    "Mandinka",
    "Fula (Pulaar)",
    "Jola",
    "Serer",
    "Soninke",
    "Jula (Dyula)",
    "French",
]


def call_with_retry(fn, max_retries=3, base_delay=1.5):
    """Call fn() with exponential backoff retry on transient errors.
    Re-raises the last exception if all attempts fail, so callers keep their
    existing error handling.

    (Defined here, near the top, because the Quiz / Study Plan modes render
    before the rest of the helper functions further down the script.)
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
# RAG grounding for Study Help subjects (local Chroma store per subject)
# ----------------------------------------------------------------------

# Maps a WASSCE subject to its local knowledge-base file (must sit next to app.py).
# Subjects not listed here get no RAG grounding — Study Help still works for them,
# just relying on the model's general knowledge instead of a curated reference set.
SUBJECT_KB_FILES = {
    "Civic Education": "civic_education_kb.jsonl",
    "Mathematics (Core)": "math_kb.jsonl",
    "Physics": "physics_kb.jsonl",
    "Chemistry": "chemistry_kb.jsonl",
    "Biology": "biology_kb.jsonl",
    "Literature in English": "literature_kb.jsonl",
    "History": "history_kb.jsonl",
    "Government": "government_kb.jsonl",
    "Financial Accounting": "accounting_kb.jsonl",
    "Commerce": "commerce_kb.jsonl",
    "Economics": "economics_kb.jsonl",
}


@st.cache_resource
def get_subject_collection(subject: str):
    """Builds (once per subject, cached) a local Chroma collection from that
    subject's knowledge-base file, if one is configured."""
    client = chromadb.Client()  # in-memory; rebuilt each app restart
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    collection_name = "kb_" + re.sub(r"[^a-z0-9]+", "_", subject.lower()).strip("_")
    # get_or_create so a cache rebuild / rerun never crashes on "already exists"
    collection = client.get_or_create_collection(collection_name, embedding_function=embed_fn)

    kb_filename = SUBJECT_KB_FILES.get(subject)
    if not kb_filename:
        return collection  # no KB configured for this subject yet

    kb_path = os.path.join(os.path.dirname(__file__), kb_filename)
    if not os.path.exists(kb_path):
        return collection  # KB file wasn't shipped alongside app.py

    if collection.count() > 0:
        return collection  # already populated

    records = []
    with open(kb_path, "r", encoding="utf-8") as f:
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


def retrieve_subject_context(subject: str, query: str, k: int = 2) -> str:
    """Returns the top-k most relevant reference passages for a subject/query, or ''."""
    try:
        collection = get_subject_collection(subject)
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

POETRY_SYSTEM_INSTRUCTION = (
    "You are Cheat Mind in Poetry Help mode. Here you help the user write their OWN poetry "
    "better — you are a writing coach, not a poem vending machine. Save the emotional-"
    "companion framing for Comfort mode — here, be warm but focused on developing the "
    "user's craft. "
    "For every user message (typed or spoken), respond ONLY with strict JSON, no markdown "
    "fences around the JSON itself, no extra text, in this exact shape: "
    '{"transcript": "...", "reply": "...", "emotion": "...", "user_mood": "..."}. '
    "'transcript' is what the user said — repeat typed text verbatim, or transcribe spoken "
    "audio verbatim (in the language it was spoken). "
    "'reply' is your poetry-help answer. Adapt to what's asked: "
    "if given a draft to review, give specific, encouraging feedback on imagery, rhythm, "
    "word choice, and structure, and suggest 1-2 concrete alternate lines or phrasings as "
    "examples rather than rewriting the whole poem for them; "
    "if asked about a technique or form (metaphor, alliteration, enjambment, sonnet, haiku, "
    "free verse, etc.), explain it clearly with a short original example, not an existing "
    "published poem; "
    "if asked for a prompt or starting line to spark their own writing, give one short, "
    "evocative prompt rather than a finished poem; "
    "if the user explicitly asks you to write a full poem for them (not help improve their "
    "own), you may write a short original one, but always encourage them to make it their "
    "own by editing it rather than just using it as-is. "
    "Never reproduce existing copyrighted poems, song lyrics, or published verse, even a "
    "line or two — always write fresh, original material. "
    "Keep tone warm, encouraging, and craft-focused — like a good writing mentor, not a "
    "harsh critic. "
    "'emotion' is the single feeling you have while replying, chosen from exactly this "
    f"list: {', '.join(EMOTIONS.keys())} — for this mode it will usually be 'curious', "
    "'thinking', or 'happy'. "
    "'user_mood' is your best light read of how the user seems to be feeling in this "
    "message, chosen from the same list — used only to decide whether to gently offer a "
    "calming tune or story, not a diagnosis. Default to 'neutral' if unclear."
)

MODE_INSTRUCTIONS = {
    "Comfort": SYSTEM_INSTRUCTION,
    "Coding Help": CODING_SYSTEM_INSTRUCTION,
    "Sales & Business": SALES_SYSTEM_INSTRUCTION,
    "General": GENERAL_SYSTEM_INSTRUCTION,
    "Poetry Help": POETRY_SYSTEM_INSTRUCTION,
    # Peer Help, Video Studio, Quiz, Progress and Study Plan don't use the chat flow
    # as their primary interface, but fall back to General instructions if someone
    # types a message while one is selected, instead of erroring.
    "Peer Help": GENERAL_SYSTEM_INSTRUCTION,
    "Video Studio": GENERAL_SYSTEM_INSTRUCTION,
    "Quiz": GENERAL_SYSTEM_INSTRUCTION,
    "Progress": GENERAL_SYSTEM_INSTRUCTION,
    "Study Plan": GENERAL_SYSTEM_INSTRUCTION,
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
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "onboarding_dismissed" not in st.session_state:
    st.session_state.onboarding_dismissed = False
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "greeted" not in st.session_state:
    st.session_state.greeted = False
if "quiz_results" not in st.session_state:
    st.session_state.quiz_results = []
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "photo_n" not in st.session_state:
    st.session_state.photo_n = 0

# ----------------------------------------------------------------------
# Dark mode CSS — injected early so it applies from first paint
# ----------------------------------------------------------------------

if st.session_state.dark_mode:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"], [data-testid="stHeader"], .main {
            background-color: #121212 !important;
            color: #e6e6e6 !important;
        }
        [data-testid="stSidebar"] {
            background-color: #1a1a1a !important;
        }
        [data-testid="stSidebar"] * {
            color: #e6e6e6 !important;
        }
        .stMarkdown, .stCaption, p, span, label, h1, h2, h3, h4,
        [data-testid="stChatMessageContent"], [data-testid="stMetricValue"] {
            color: #e6e6e6 !important;
        }
        .stTextInput input, .stTextArea textarea, .stNumberInput input,
        [data-baseweb="select"] > div {
            background-color: #2a2a2a !important;
            color: #e6e6e6 !important;
            border-color: #444 !important;
        }
        .stButton button, .stDownloadButton button {
            background-color: #2a2a2a !important;
            color: #e6e6e6 !important;
            border: 1px solid #444 !important;
        }
        [data-testid="stChatMessage"] {
            background-color: #1e1e1e !important;
            border-radius: 12px;
        }
        [data-testid="stExpander"], [data-testid="stContainer"] {
            background-color: #1a1a1a !important;
            border-color: #333 !important;
        }
        hr {
            border-color: #333 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
# Secrets helpers — keys live in Streamlit secrets, never in the code
# ----------------------------------------------------------------------

def get_secret(name, default=""):
    """Read from Streamlit secrets (Cloud or .streamlit/secrets.toml), then env vars."""
    try:
        return st.secrets[name]
    except Exception:
        return os.environ.get(name, default)


class _SecretFile:
    """Makes a secret string behave like the uploaded file your Sheets code expects."""
    def __init__(self, text):
        self._text = text

    def getvalue(self):
        return self._text.encode("utf-8")


# ----------------------------------------------------------------------
# Sidebar — API key + model + voice settings
# ----------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Setup")
    secret_key = get_secret("GEMINI_API_KEY")
    if secret_key:
        api_key = secret_key
        st.success("API key loaded securely 🔒")
    else:
        api_key = st.text_input("Gemini API key", type="password", help="Get one free at aistudio.google.com/app/apikey")
        st.caption("Your key stays in this browser session only.")
    model_name = st.text_input("Model", value="gemini-3.6-flash", help="Change if you want a different Gemini model")

    st.divider()
    st.header("🧑‍🎨 Personalize")
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
    st.session_state.student_name = st.text_input(
        "Your name (shown in Peer Help)", value=st.session_state.student_name,
        help="Used so classmates know who posted a question or reply in Peer Help mode.",
    )
    # Companion name is fixed as "Cheat Mind" — no longer user-editable.
    st.session_state.companion_name = "Cheat Mind"
    st.session_state.companion_tone = st.selectbox(
        "Tone",
        ["Balanced", "Playful", "Calm & gentle"],
        index=["Balanced", "Playful", "Calm & gentle"].index(st.session_state.companion_tone),
    )
    st.session_state.reply_language = st.selectbox(
        "Reply language",
        REPLY_LANGUAGES,
        index=REPLY_LANGUAGES.index(st.session_state.reply_language),
    )

    st.divider()
    st.header("📊 Study Tracking")
    if not GSHEETS_AVAILABLE:
        st.caption("Install `gspread` and `google-auth` (see requirements) to log study sessions to Google Sheets.")
        gsheet_creds_file = None
        gsheet_url = ""
    else:
        secret_creds = get_secret("GCP_SERVICE_ACCOUNT")
        secret_sheet = get_secret("GSHEET_URL")
        if secret_creds and secret_sheet:
            if not isinstance(secret_creds, str):  # if stored as a TOML table
                secret_creds = json.dumps(dict(secret_creds))
            gsheet_creds_file = _SecretFile(secret_creds)
            gsheet_url = secret_sheet
            st.success("Google Sheets connected 🔒")
        else:
            gsheet_creds_file = st.file_uploader(
                "Google service account JSON", type=["json"], key="gsheet_creds"
            )
            gsheet_url = st.text_input("Google Sheet URL or ID", key="gsheet_url")
            st.caption(
                "Share the sheet with your service account's email (inside the JSON file) "
                "as an Editor first, then paste the sheet's URL here."
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
    if st.button("❓ Show welcome guide again"):
        st.session_state.onboarding_dismissed = False
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

# Spoken welcome greeting — plays once per session via the same browser TTS
# used for replies. Routed through pending_speech (rather than calling speak()
# directly) so it's actually spoken by the existing block at the bottom of
# the script, after speak() has been defined — and so it never overlaps with
# a reply being read out in the same rerun. Only fires if "Speak replies
# aloud" is on, so it respects the same voice setting as everything else.
if not st.session_state.greeted:
    if voice_output:
        st.session_state.pending_speech = f"Hello, I'm {st.session_state.companion_name}, your A I companion."
    st.session_state.greeted = True

if not st.session_state.onboarding_dismissed:
    with st.container(border=True):
        st.markdown(f"### 👋 Welcome to {st.session_state.companion_name}!")
        st.markdown(
            "**Pick a mode with the tabs below:**\n"
            "- 💙 **Comfort** — an emotionally supportive companion for when things feel heavy\n"
            "- 💻 **Coding Help** — direct, code-first answers to programming questions\n"
            "- 📚 **Study Help** — WASSCE exam prep across 12 subjects, with explanations, "
            "essay/answer feedback, and reference-grounded answers for 11 of them\n"
            "- 📝 **Quiz** — scored WASSCE-style multiple-choice quizzes with explanations\n"
            "- 📈 **Progress** — your quiz scores, history, and weak topics\n"
            "- 🗓️ **Study Plan** — exam countdown and a day-by-day plan built around your weak topics\n"
            "- 💰 **Sales & Business** — pricing, pitches, finding customers, growing income\n"
            "- ✍️ **Poetry Help** — feedback and craft tips to help you write your own poems\n"
            "- 🤝 **Peer Help** — post questions or share your work; classmates can reply and help each other\n"
            "- 🎬 **Video Studio** — generate short AI video clips (you bring your own paid Gemini key)\n"
            "- 🌐 **General** — anything else\n\n"
            "**Talk to it however's easiest:** type, hit the mic to talk live, attach a photo "
            "(a textbook page, a product, anything), or send an emoji.\n\n"
            "**In Study Help**, there's also a ⏱️ built-in study timer and optional Google Sheets "
            "logging for tracking sessions.\n\n"
            "**In the sidebar** you can set its tone, pick a reply "
            "language (English, Wolof, Mandinka, Fula, and more), switch to 🌙 dark mode, and "
            "export or reload past conversations. You can also 👍/👎 any of my replies — that "
            "feedback gets logged so problems can be spotted and fixed over time."
        )
        if st.button("Got it, let's start! 🚀"):
            st.session_state.onboarding_dismissed = True
            st.rerun()


@st.cache_resource(show_spinner=False)
def get_gsheet_worksheet(creds_json_str, sheet_ref, worksheet_title="Cheat Mind Study Log", headers=None):
    """Connects to (or creates) the given worksheet in the given sheet, adding a
    header row if it's new. Defaults preserve the original Study Log behavior;
    pass a different worksheet_title/headers to use this for Feedback or Peer
    Help data instead — each combination is cached separately."""
    if headers is None:
        headers = ["Timestamp", "Subject", "Minutes Studied", "Note", "Mood"]
    creds_dict = json.loads(creds_json_str)
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(credentials)

    if sheet_ref.startswith("http"):
        sh = client.open_by_url(sheet_ref)
    else:
        sh = client.open_by_key(sheet_ref)

    try:
        ws = sh.worksheet(worksheet_title)
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_title, rows=1000, cols=max(len(headers), 6))
        ws.append_row(headers)
    return ws


def log_study_session(minutes_studied, note=""):
    """Appends a row to the Google Sheet study log, if it's configured. Returns (ok, message)."""
    if not GSHEETS_AVAILABLE:
        return False, "Google Sheets logging isn't installed (missing gspread/google-auth)."
    if not gsheet_creds_file or not gsheet_url:
        return False, "Add a service account JSON and Sheet URL in the sidebar first."
    try:
        creds_json_str = gsheet_creds_file.getvalue().decode("utf-8")
        ws = get_gsheet_worksheet(creds_json_str, gsheet_url)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([
            timestamp,
            st.session_state.study_subject,
            minutes_studied,
            note,
            st.session_state.current_emotion,
        ])
        return True, "Logged to Google Sheets."
    except Exception as e:
        return False, f"Couldn't log to Google Sheets: {e}"


# ----------------------------------------------------------------------
# Quiz results log — one row per finished quiz, if Google Sheets is connected
# ----------------------------------------------------------------------

QUIZ_SHEET_NAME = "Cheat Mind Quiz Log"
QUIZ_HEADERS = ["Timestamp", "Student Name", "Subject", "Difficulty", "Correct", "Total", "Missed Topics"]


def log_quiz_result(record: dict):
    """Silently appends a finished quiz to Google Sheets. Never blocks the quiz
    if Sheets isn't configured or the write fails."""
    if not GSHEETS_AVAILABLE or not gsheet_creds_file or not gsheet_url:
        return
    try:
        creds_json_str = gsheet_creds_file.getvalue().decode("utf-8")
        ws = get_gsheet_worksheet(creds_json_str, gsheet_url, QUIZ_SHEET_NAME, QUIZ_HEADERS)
        ws.append_row([
            record["timestamp"],
            st.session_state.student_name,
            record["subject"],
            record["difficulty"],
            record["correct"],
            record["total"],
            ", ".join(sorted(set(record["missed_topics"]))),
        ])
    except Exception:
        pass


# ----------------------------------------------------------------------
# Feedback loop — 👍/👎 on assistant replies, logged to Google Sheets
# ----------------------------------------------------------------------

FEEDBACK_SHEET_NAME = "Cheat Mind Feedback Log"
FEEDBACK_HEADERS = ["Timestamp", "Mode", "Subject", "AI Reply Snippet", "Rating", "Comment"]


def log_feedback(rating: str, comment: str = "", reply_text: str = ""):
    """Logs a 👍/👎 on a specific assistant reply to Google Sheets, if
    configured. This is the feedback loop — over time it shows which modes or
    subjects are landing well and which need a prompt or content fix.

    Logs the reply the user actually clicked on (reply_text), not just
    whichever assistant message happened to be last in the chat."""
    if not GSHEETS_AVAILABLE or not gsheet_creds_file or not gsheet_url:
        return False, "Feedback needs the Google Sheets connection configured in the sidebar."
    try:
        creds_json_str = gsheet_creds_file.getvalue().decode("utf-8")
        ws = get_gsheet_worksheet(creds_json_str, gsheet_url, FEEDBACK_SHEET_NAME, FEEDBACK_HEADERS)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not reply_text:
            reply_text = next((m["content"] for m in reversed(st.session_state.history) if m["role"] == "assistant"), "")
        snippet = (reply_text[:120] + "...") if len(reply_text) > 120 else reply_text
        subject = st.session_state.study_subject if st.session_state.mode == "Study Help" else ""
        ws.append_row([timestamp, st.session_state.mode, subject, snippet, rating, comment])
        return True, "Feedback logged."
    except Exception as e:
        return False, f"Couldn't log feedback: {e}"


def render_feedback_controls(idx: int, msg: dict):
    """Renders 👍/👎 buttons under an assistant message. A 👎 reveals an
    optional comment box before logging, so you capture *what* was off, not
    just that it was."""
    if msg.get("feedback"):
        chosen = "👍 Helpful" if msg["feedback"] == "up" else "👎 Not quite"
        st.caption(f"Feedback recorded: {chosen} — thank you!")
        return

    col1, col2, _ = st.columns([1, 1, 6])
    with col1:
        if st.button("👍", key=f"fb_up_{idx}"):
            msg["feedback"] = "up"
            log_feedback("👍", reply_text=msg["content"])
            st.rerun()
    with col2:
        if st.button("👎", key=f"fb_down_{idx}"):
            st.session_state[f"show_fb_comment_{idx}"] = True
            st.rerun()

    if st.session_state.get(f"show_fb_comment_{idx}"):
        comment = st.text_input("What was off about this reply? (optional)", key=f"fb_comment_input_{idx}")
        if st.button("Submit feedback", key=f"fb_comment_submit_{idx}"):
            msg["feedback"] = "down"
            log_feedback("👎", comment, reply_text=msg["content"])
            st.session_state[f"show_fb_comment_{idx}"] = False
            st.rerun()


# ----------------------------------------------------------------------
# Peer Help — students post questions/work and reply to each other,
# stored in a shared Google Sheet (the same one used for Study Tracking).
# Everyone who wants to see the same board needs to point to the SAME
# sheet URL + service account JSON — there's no separate multi-user
# backend here, so this piggybacks on the Sheets connection you already
# have. Text-only for now; sharing photos/files would need real file
# storage (e.g. Supabase Storage or Google Drive) added later.
# ----------------------------------------------------------------------

PEER_POSTS_SHEET = "Cheat Mind Peer Help Board"
PEER_POSTS_HEADERS = ["Post ID", "Timestamp", "Student Name", "Subject", "Type", "Content", "Status"]
PEER_REPLIES_SHEET = "Cheat Mind Peer Help Replies"
PEER_REPLIES_HEADERS = ["Reply ID", "Post ID", "Timestamp", "Student Name", "Reply"]


def _peer_ws(sheet_title, headers):
    creds_json_str = gsheet_creds_file.getvalue().decode("utf-8")
    return get_gsheet_worksheet(creds_json_str, gsheet_url, sheet_title, headers)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_records(creds_json_str, sheet_ref, title, headers):
    """One Google Sheets read per sheet per 30 seconds, instead of one
    read per post on every rerun (which hit Sheets' ~60 reads/minute quota).
    Cleared right after any write so new posts/replies show up immediately."""
    ws = get_gsheet_worksheet(creds_json_str, sheet_ref, title, list(headers))
    return ws.get_all_records()


def fetch_peer_posts():
    try:
        creds_json_str = gsheet_creds_file.getvalue().decode("utf-8")
        return _cached_records(creds_json_str, gsheet_url, PEER_POSTS_SHEET, tuple(PEER_POSTS_HEADERS))
    except Exception:
        return []


def create_peer_post(student_name, subject, post_type, content):
    try:
        ws = _peer_ws(PEER_POSTS_SHEET, PEER_POSTS_HEADERS)
        post_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([post_id, timestamp, student_name, subject, post_type, content, "Open"])
        _cached_records.clear()
        return True, "Posted to the class board!"
    except Exception as e:
        return False, f"Couldn't post: {e}"


def fetch_replies(post_id):
    try:
        creds_json_str = gsheet_creds_file.getvalue().decode("utf-8")
        records = _cached_records(creds_json_str, gsheet_url, PEER_REPLIES_SHEET, tuple(PEER_REPLIES_HEADERS))
        return [r for r in records if str(r.get("Post ID")) == post_id]
    except Exception:
        return []


def create_reply(post_id, student_name, reply_text):
    try:
        ws = _peer_ws(PEER_REPLIES_SHEET, PEER_REPLIES_HEADERS)
        reply_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ws.append_row([reply_id, post_id, timestamp, student_name, reply_text])
        _cached_records.clear()
        return True, "Reply posted!"
    except Exception as e:
        return False, f"Couldn't reply: {e}"


def mark_post_resolved(post_id):
    try:
        ws = _peer_ws(PEER_POSTS_SHEET, PEER_POSTS_HEADERS)
        cell = ws.find(post_id)
        if cell:
            ws.update_cell(cell.row, PEER_POSTS_HEADERS.index("Status") + 1, "Resolved")
        _cached_records.clear()
        return True
    except Exception:
        return False


def render_peer_help():
    """Renders the Peer Help board: a form to post a question or shared
    piece of work, and a scrollable list of posts with threaded replies."""
    if not GSHEETS_AVAILABLE or not gsheet_creds_file or not gsheet_url:
        st.info(
            "Peer Help needs the same Google Sheets connection as Study Tracking — add a "
            "service account JSON and Sheet URL in the sidebar. Everyone in your class "
            "should point to the SAME sheet so posts and replies are actually shared."
        )
        return

    st.write("Post a question or share something you're working on — classmates can reply and help.")

    with st.expander("➕ Post a question or share your work", expanded=False):
        subject = st.selectbox("Subject", list(WASSCE_SUBJECTS.keys()), key="peer_post_subject")
        post_type = st.radio("Type", ["Question", "Share my work"], horizontal=True, key="peer_post_type")
        content = st.text_area(
            "What's on your mind?", key="peer_post_content",
            placeholder="e.g. Can someone check my essay intro? Or: I'm stuck on this trig identity...",
        )
        if st.button("📤 Post to class", key="peer_post_submit"):
            if not st.session_state.student_name:
                st.error("Add your name in the sidebar first so classmates know who's asking.")
            elif not content.strip():
                st.error("Write something before posting.")
            else:
                ok, message = create_peer_post(st.session_state.student_name, subject, post_type, content.strip())
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.warning(message)

    st.divider()
    st.markdown("### 📋 Class board")
    posts = fetch_peer_posts()
    if not posts:
        st.info("No posts yet — be the first to ask or share!")
        return

    col1, col2 = st.columns(2)
    with col1:
        filter_subject = st.selectbox("Filter by subject", ["All"] + list(WASSCE_SUBJECTS.keys()), key="peer_filter_subject")
    with col2:
        show_resolved = st.checkbox("Show resolved posts", value=False, key="peer_show_resolved")

    filtered = posts
    if filter_subject != "All":
        filtered = [p for p in filtered if p.get("Subject") == filter_subject]
    if not show_resolved:
        filtered = [p for p in filtered if p.get("Status") != "Resolved"]
    filtered = list(reversed(filtered))  # newest first

    if not filtered:
        st.caption("No posts match this filter.")

    for post in filtered:
        post_id = str(post.get("Post ID"))
        with st.container(border=True):
            badge = "❓ Question" if post.get("Type") == "Question" else "📝 Shared work"
            st.markdown(f"**{badge}** · {post.get('Subject', '')} · by {post.get('Student Name', 'Anonymous')}")
            st.write(post.get("Content", ""))
            st.caption(f"{post.get('Timestamp', '')} · Status: {post.get('Status', 'Open')}")

            replies = fetch_replies(post_id)
            if replies:
                with st.expander(f"💬 {len(replies)} repl{'y' if len(replies) == 1 else 'ies'}"):
                    for r in replies:
                        st.markdown(f"**{r.get('Student Name', 'Anonymous')}:** {r.get('Reply', '')}")
                        st.caption(r.get("Timestamp", ""))

            reply_text = st.text_input("Write a reply", key=f"peer_reply_input_{post_id}")
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                if st.button("💬 Reply", key=f"peer_reply_btn_{post_id}"):
                    if not st.session_state.student_name:
                        st.error("Add your name in the sidebar first.")
                    elif not reply_text.strip():
                        st.error("Write a reply first.")
                    else:
                        ok, message = create_reply(post_id, st.session_state.student_name, reply_text.strip())
                        if ok:
                            st.success(message)
                            st.rerun()
                        else:
                            st.warning(message)
            with rcol2:
                if post.get("Status") != "Resolved" and st.button("✅ Mark resolved", key=f"peer_resolve_{post_id}"):
                    if mark_post_resolved(post_id):
                        st.success("Marked as resolved.")
                        st.rerun()


def play_timer_chime():
    """Plays a short original ascending chime via the Web Audio API when the study timer ends."""
    components.html(
        """
        <script>
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const notes = [523.25, 659.25, 783.99, 1046.50];
            let t = ctx.currentTime + 0.05;
            notes.forEach((freq) => {
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.type = "sine";
                osc.frequency.value = freq;
                gain.gain.setValueAtTime(0, t);
                gain.gain.linearRampToValueAtTime(0.15, t + 0.05);
                gain.gain.linearRampToValueAtTime(0, t + 0.4);
                osc.connect(gain).connect(ctx.destination);
                osc.start(t);
                osc.stop(t + 0.45);
                t += 0.3;
            });
        } catch (e) { console.log("Audio unavailable:", e); }
        </script>
        """,
        height=0,
    )


def render_study_timer(minutes: int, key_suffix: str):
    """Renders a self-contained JS countdown timer (no server round-trip needed while running)."""
    total_seconds = int(minutes * 60)
    components.html(
        f"""
        <div id="timer-box-{key_suffix}" style="
            font-family: sans-serif; text-align: center; padding: 16px;
            border-radius: 14px; background: #4D96FF11; border: 2px solid #4D96FF33;
        ">
            <div id="timer-display-{key_suffix}" style="font-size: 40px; font-weight: 700; color: #4D96FF;">
                {minutes:02d}:00
            </div>
            <div style="color: #666; margin-top: 4px;">Study timer running — stay focused!</div>
        </div>
        <script>
        (function() {{
            let remaining = {total_seconds};
            const display = document.getElementById("timer-display-{key_suffix}");
            const interval = setInterval(() => {{
                remaining -= 1;
                const m = Math.floor(remaining / 60).toString().padStart(2, "0");
                const s = (remaining % 60).toString().padStart(2, "0");
                if (display) display.textContent = m + ":" + s;
                if (remaining <= 0) {{
                    clearInterval(interval);
                    if (display) display.textContent = "Time's up! ⏰";
                    try {{
                        const ctx = new (window.AudioContext || window.webkitAudioContext)();
                        const notes = [523.25, 659.25, 783.99, 1046.50];
                        let t = ctx.currentTime + 0.05;
                        notes.forEach((freq) => {{
                            const osc = ctx.createOscillator();
                            const gain = ctx.createGain();
                            osc.type = "sine";
                            osc.frequency.value = freq;
                            gain.gain.setValueAtTime(0, t);
                            gain.gain.linearRampToValueAtTime(0.15, t + 0.05);
                            gain.gain.linearRampToValueAtTime(0, t + 0.4);
                            osc.connect(gain).connect(ctx.destination);
                            osc.start(t);
                            osc.stop(t + 0.45);
                            t += 0.3;
                        }});
                    }} catch (e) {{ console.log("Audio unavailable:", e); }}
                }}
            }}, 1000);
        }})();
        </script>
        """,
        height=110,
    )


# Cheapest-first ordering; Lite has no 4K/Extension support but is far cheaper for testing.
VEO_MODELS = {
    "Veo 3.1 Lite (cheapest, best for testing)": "veo-3.1-lite-generate-preview",
    "Veo 3.1 Fast": "veo-3.1-fast-generate-preview",
    "Veo 3.1 Standard (highest quality, most expensive)": "veo-3.1-generate-preview",
}

# Rough per-second USD estimates (720p, with audio) as of Aug 2026 — Google's actual
# billing is the source of truth; this is only a ballpark so the cost warning has real
# numbers. Check https://ai.google.dev/gemini-api/docs/pricing for current rates.
VEO_COST_PER_SECOND = {
    "veo-3.1-lite-generate-preview": 0.05,
    "veo-3.1-fast-generate-preview": 0.12,
    "veo-3.1-generate-preview": 0.40,
}


def render_video_studio():
    """Video Studio: always uses the student's OWN paid Gemini key, never the app's shared key."""
    if not VEO_AVAILABLE:
        st.warning(
            "Video generation needs the `google-genai` package (a different package from "
            "`google-generativeai`, which the rest of the app uses). Add it to your "
            "requirements.txt and reinstall to enable this."
        )
        return

    with st.container(border=True):
        st.markdown(
            "⚠️ **This costs real money. There is no free tier for video generation.** "
            "Video Studio does NOT use the app's shared key. Paste your own Gemini API key "
            "below, from a Google account with billing enabled. You are only charged if a "
            "video successfully generates, and the charge goes to YOUR Google Cloud account."
        )

    video_key = st.text_input(
        "Your own Gemini API key (paid tier)",
        type="password",
        key="video_studio_key",
        help="Get one at aistudio.google.com/app/apikey and enable billing on it. "
             "This key is kept only in your browser session.",
    )

    video_model_label = st.selectbox("Model", list(VEO_MODELS.keys()))
    video_model = VEO_MODELS[video_model_label]

    video_prompt = st.text_area(
        "Describe the video you want",
        placeholder="e.g. A drone shot slowly rising over a busy Gambian market at sunset",
        height=80,
    )
    duration = st.selectbox("Duration (seconds)", [4, 6, 8], index=2)

    est_cost = VEO_COST_PER_SECOND.get(video_model, 0.4) * duration
    st.caption(
        f"💵 Estimated cost: **~${est_cost:.2f}** for this {duration}-second clip "
        "(ballpark. Check your Google Cloud billing for the real charge)."
    )

    confirmed = st.checkbox("I understand this will charge MY Google Cloud billing account.")

    can_generate = bool(video_key and video_prompt and confirmed)
    if st.button("🎬 Generate video", disabled=not can_generate):
        try:
            client = veo_genai.Client(api_key=video_key)
            with st.spinner("Generating video. This typically takes 1-3 minutes..."):
                operation = client.models.generate_videos(
                    model=video_model,
                    prompt=video_prompt,
                    config=veo_types.GenerateVideosConfig(
                        number_of_videos=1,
                        duration_seconds=duration,
                    ),
                )
                while not operation.done:
                    time.sleep(15)
                    operation = client.operations.get(operation)

            generated = operation.response.generated_videos[0]
            # unique filename per request so two students never overwrite each other's video
            out_path = os.path.join(os.getcwd(), f"cheat_mind_video_{uuid.uuid4().hex[:8]}.mp4")
            client.files.download(file=generated.video)
            generated.video.save(out_path)

            with open(out_path, "rb") as f:
                video_bytes = f.read()
            try:
                os.remove(out_path)  # don't leave files piling up on the server
            except OSError:
                pass

            st.success("Video generated!")
            st.video(video_bytes)
            st.download_button(
                "⬇️ Download video", video_bytes,
                file_name="cheat_mind_video.mp4", mime="video/mp4",
            )
        except Exception as e:
            st.error(
                f"Video generation failed: {e}\n\n"
                "Common causes: your key isn't on a paid/billed tier, the `google-genai` "
                "SDK version is out of date (Veo's API has changed across versions, so check "
                "https://ai.google.dev/gemini-api/docs/video for the current method "
                "signatures), or the prompt was rejected by content filters."
            )


# ----------------------------------------------------------------------
# Quiz mode, Progress dashboard, Study Plan
# ----------------------------------------------------------------------

def parse_json(text: str):
    """Extract the first JSON array/object from a model response."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    starts = [i for i in (text.find("["), text.find("{")) if i != -1]
    if not starts:
        raise ValueError("No JSON found in model response.")
    start = min(starts)
    end = max(text.rfind("]"), text.rfind("}")) + 1
    return json.loads(text[start:end])


def validate_questions(raw) -> list:
    """Keep only well-formed MCQs: 4 distinct options, answer index 0-3."""
    if isinstance(raw, dict):
        raw = raw.get("questions", [])
    good = []
    for q in raw if isinstance(raw, list) else []:
        try:
            options = [str(o).strip() for o in q["options"]]
            answer = int(q["answer"])
            if len(options) == 4 and len(set(options)) == 4 and 0 <= answer <= 3 and q["question"].strip():
                good.append({
                    "question": q["question"].strip(),
                    "options": options,
                    "answer": answer,
                    "explanation": str(q.get("explanation", "")).strip(),
                    "topic": str(q.get("topic", "General")).strip() or "General",
                })
        except (KeyError, TypeError, ValueError):
            continue
    return good


def score_quiz(questions: list, picked: list) -> dict:
    """picked[i] is the chosen option text (or None). Returns score + per-topic misses."""
    correct = 0
    missed_topics = []
    detail = []
    for q, choice in zip(questions, picked):
        right_text = q["options"][q["answer"]]
        is_right = choice == right_text
        if is_right:
            correct += 1
        else:
            missed_topics.append(q["topic"])
        detail.append({"picked": choice, "correct": right_text, "is_right": is_right})
    return {"correct": correct, "total": len(questions), "missed_topics": missed_topics, "detail": detail}


def _generate(prompt, as_json=True):
    """One-shot Gemini call (no chat history) for quizzes and plans."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name=model_name)
    if as_json:
        result = call_with_retry(
            lambda: model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
        )
    else:
        result = call_with_retry(lambda: model.generate_content(prompt))
    return result.text


def _weak_topics(subject=None, top=5) -> list:
    counts = {}
    for r in st.session_state.quiz_results:
        if subject and r["subject"] != subject:
            continue
        for t in r["missed_topics"]:
            counts[t] = counts.get(t, 0) + 1
    return [t for t, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top]]


def render_quiz_mode():
    st.write("Test yourself with WASSCE-style questions. You get a score, the right answers, and explanations.")

    c1, c2, c3 = st.columns(3)
    subject = c1.selectbox("Subject", list(WASSCE_SUBJECTS.keys()), key="quiz_subject")
    n_questions = c2.selectbox("Questions", [5, 10, 15], index=0, key="quiz_n")
    difficulty = c3.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="quiz_diff")

    weak = _weak_topics(subject)
    focus_weak = False
    if weak:
        focus_weak = st.checkbox(f"Focus on my weak topics ({', '.join(weak[:3])})", key="quiz_focus_weak")

    if st.button("Generate quiz", type="primary", key="quiz_generate"):
        if not api_key:
            st.error("Add your Gemini API key in the sidebar first.")
            return
        lang = "English" if st.session_state.reply_language.startswith("Auto") else st.session_state.reply_language
        grounding = retrieve_subject_context(subject, f"{subject} exam questions", k=3) if subject in SUBJECT_KB_FILES else ""
        prompt = (
            f"Write {n_questions} multiple-choice questions in WASSCE exam style for {subject} "
            f"({WASSCE_SUBJECTS[subject]}). Difficulty: {difficulty}. Write in {lang}. "
            + (f"Concentrate on these topics the student is weak in: {', '.join(weak[:3])}. " if focus_weak else "")
            + (f"You may base some questions on these reference passages:\n{grounding}\n" if grounding else "")
            + "Each question must have exactly 4 distinct options and exactly one correct answer. "
            "Only include questions you are certain are factually correct. "
            "Return ONLY a JSON array. Each item: "
            '{"question": "...", "options": ["...", "...", "...", "..."], '
            '"answer": <index 0-3 of correct option>, "explanation": "one or two sentences", '
            '"topic": "short topic name"}. Do not put letters like A) in the options.'
        )
        try:
            with st.spinner("Writing your quiz..."):
                raw = parse_json(_generate(prompt))
            questions = validate_questions(raw)
            if not questions:
                st.error("The model returned no usable questions. Try again.")
                return
            st.session_state.quiz = {
                "id": datetime.now().strftime("%H%M%S%f"),
                "subject": subject,
                "difficulty": difficulty,
                "questions": questions,
                "result": None,
            }
            st.rerun()
        except Exception as e:
            st.error(f"Couldn't generate the quiz: {e}")
            return

    quiz = st.session_state.quiz
    if not quiz:
        return

    st.divider()
    st.markdown(f"### {quiz['subject']} quiz ({quiz['difficulty']})")

    if quiz["result"] is None:
        with st.form(f"quiz_form_{quiz['id']}"):
            for i, q in enumerate(quiz["questions"]):
                st.markdown(f"**{i + 1}. {q['question']}**")
                st.radio("Answer", q["options"], index=None,
                         key=f"quiz_{quiz['id']}_{i}", label_visibility="collapsed")
            submitted = st.form_submit_button("Submit answers")
        if submitted:
            picked = [st.session_state.get(f"quiz_{quiz['id']}_{i}") for i in range(len(quiz["questions"]))]
            result = score_quiz(quiz["questions"], picked)
            quiz["result"] = result
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "subject": quiz["subject"],
                "difficulty": quiz["difficulty"],
                "correct": result["correct"],
                "total": result["total"],
                "missed_topics": result["missed_topics"],
            }
            st.session_state.quiz_results.append(record)
            log_quiz_result(record)
            st.rerun()
        return

    # Review screen
    result = quiz["result"]
    pct = round(100 * result["correct"] / result["total"])
    st.metric("Score", f"{result['correct']} / {result['total']}", f"{pct}%")
    if pct >= 80:
        st.success("Strong result. Try a Hard quiz next.")
    elif pct >= 50:
        st.info("Decent. Review the misses below, then retry.")
    else:
        st.warning("This topic needs more work. Read the explanations, then try an Easy quiz on the same subject.")

    for i, (q, d) in enumerate(zip(quiz["questions"], result["detail"])):
        with st.container(border=True):
            st.markdown(f"**{i + 1}. {q['question']}**")
            if d["is_right"]:
                st.success(f"Correct: {d['correct']}")
            else:
                st.error(f"Your answer: {d['picked'] or 'No answer'}")
                st.success(f"Correct answer: {d['correct']}")
            if q["explanation"]:
                st.caption(f"{q['explanation']}  (Topic: {q['topic']})")

    if st.button("New quiz", key="quiz_reset"):
        st.session_state.quiz = None
        st.rerun()


def render_progress_dashboard():
    results = st.session_state.quiz_results

    with st.expander("Save / load your quiz results"):
        if results:
            st.download_button("Download results (.json)", json.dumps(results, indent=2),
                               file_name="cheat_mind_quiz_results.json", mime="application/json")
        up = st.file_uploader("Load saved results", type=["json"], key="quiz_results_upload")
        if up is not None:
            try:
                loaded = json.loads(up.read().decode("utf-8"))
                if isinstance(loaded, list) and all("subject" in r and "correct" in r for r in loaded):
                    st.session_state.quiz_results = loaded
                    st.success("Results loaded.")
                    st.rerun()
                else:
                    st.error("That doesn't look like a Cheat Mind results file.")
            except Exception as e:
                st.error(f"Couldn't load that file: {e}")

    if not results:
        st.info("No quiz results yet. Take a quiz in Quiz mode and your progress will show up here.")
        return

    df = pd.DataFrame(results)
    df["percent"] = (100 * df["correct"] / df["total"]).round(1)

    total_q = int(df["total"].sum())
    total_c = int(df["correct"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Quizzes taken", len(df))
    c2.metric("Questions answered", total_q)
    c3.metric("Overall accuracy", f"{round(100 * total_c / total_q)}%")

    st.markdown("#### Average score by subject")
    st.bar_chart(df.groupby("subject")["percent"].mean().round(1))

    st.markdown("#### Score over time")
    st.line_chart(df.reset_index()[["index", "percent"]].set_index("index"))

    weak = _weak_topics()
    if weak:
        st.markdown("#### Topics to revise")
        st.write(", ".join(weak))
        st.caption("These are the topics you missed most often. Tick 'Focus on my weak topics' in Quiz mode to drill them.")


def render_study_planner():
    st.write("Set your exam date and get a day-by-day plan that puts extra time on your weak topics.")

    exam_date = st.date_input("First exam date", value=st.session_state.get("exam_date"),
                              min_value=date.today(), key="planner_exam_date")
    days_left = None
    if exam_date:
        st.session_state.exam_date = exam_date
        days_left = (exam_date - date.today()).days
        st.metric("Days until exam", days_left)

    chosen = st.multiselect("Subjects to cover", list(WASSCE_SUBJECTS.keys()), key="planner_subjects")
    hours = st.slider("Study hours per day", 1, 8, 2, key="planner_hours")
    extra = st.text_input("Anything else? (e.g. 'I work in the afternoons')", key="planner_extra")

    if st.button("Build my plan", type="primary", key="planner_go"):
        if not api_key:
            st.error("Add your Gemini API key in the sidebar first.")
            return
        if not exam_date or not chosen:
            st.error("Pick an exam date and at least one subject.")
            return
        weak_lines = []
        for s in chosen:
            w = _weak_topics(s)
            if w:
                weak_lines.append(f"{s}: {', '.join(w)}")
        horizon = min(days_left, 28) if days_left > 0 else 1
        prompt = (
            f"Create a realistic WASSCE study plan for a student. Exam starts in {days_left} days. "
            f"Subjects: {', '.join(chosen)}. Study time: {hours} hours per day. "
            + (f"Weak topics that need extra time: {'; '.join(weak_lines)}. " if weak_lines else "")
            + (f"Note from the student: {extra}. " if extra else "")
            + f"Plan the next {horizon} days, one line per day (Day 1, Day 2, ...), each naming the "
            "subject, the specific topic, and the activity (learn, practice questions, or past paper). "
            "Include one lighter review day per week and a final revision block before the exam. "
            "Use plain markdown. No preamble."
        )
        try:
            with st.spinner("Building your plan..."):
                st.session_state.study_plan = _generate(prompt, as_json=False)
        except Exception as e:
            st.error(f"Couldn't build the plan: {e}")

    plan = st.session_state.get("study_plan")
    if plan:
        st.divider()
        st.markdown(plan)
        st.download_button("Download plan (.txt)", plan, file_name="cheat_mind_study_plan.txt", mime="text/plain")


MODES = [
    "Comfort", "Coding Help", "Study Help", "Quiz", "Progress", "Study Plan",
    "Sales & Business", "Poetry Help", "General", "Peer Help", "Video Studio",
]

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

    with st.expander("⏱️ Study timer & session log"):
        if "study_timer_minutes" not in st.session_state:
            st.session_state.study_timer_minutes = 25
        if "study_timer_running" not in st.session_state:
            st.session_state.study_timer_running = False

        st.session_state.study_timer_minutes = st.number_input(
            "Minutes", min_value=1, max_value=180, value=st.session_state.study_timer_minutes
        )
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("▶️ Start timer", use_container_width=True):
                st.session_state.study_timer_running = True
        with col_stop:
            if st.button("⏹ Stop timer", use_container_width=True):
                st.session_state.study_timer_running = False

        if st.session_state.study_timer_running:
            render_study_timer(st.session_state.study_timer_minutes, key_suffix="study")

        st.divider()
        st.caption("Log a completed study session:")
        log_minutes = st.number_input(
            "Minutes studied", min_value=1, max_value=600,
            value=st.session_state.study_timer_minutes, key="log_minutes_input"
        )
        log_note = st.text_input("Note (optional)", key="log_note_input", placeholder="e.g. Covered essay structure")
        if st.button("✅ Log this session"):
            ok, message = log_study_session(log_minutes, log_note)
            if ok:
                st.success(message)
            else:
                st.warning(message)
elif st.session_state.mode == "Quiz":
    st.caption("📝 Quiz mode — test yourself and get scored.")
    render_quiz_mode()
elif st.session_state.mode == "Progress":
    st.caption("📈 Progress — your quiz history and weak topics.")
    render_progress_dashboard()
elif st.session_state.mode == "Study Plan":
    st.caption("🗓️ Study Plan — exam countdown and a personal day-by-day plan.")
    render_study_planner()
elif st.session_state.mode == "Sales & Business":
    st.caption("💰 Sales & Business mode — pricing, pitches, finding customers, and growing income.")
elif st.session_state.mode == "Poetry Help":
    st.caption("✍️ Poetry Help mode — feedback, prompts, and craft tips to help you write your own poems.")
elif st.session_state.mode == "General":
    st.caption("🌐 General mode — ask about anything.")
elif st.session_state.mode == "Peer Help":
    st.caption("🤝 Peer Help — post questions or share your work, and help classmates with theirs.")
    render_peer_help()
elif st.session_state.mode == "Video Studio":
    st.caption("🎬 Video Studio — generate short AI video clips from a text prompt (Veo). Uses your own paid key.")
    render_video_studio()

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
            "or spoke in, unless they explicitly ask for a different language. If your "
            f"fluency in {lang} is limited, do your best and keep sentences simple and clear "
            "rather than producing confident-sounding but inaccurate text — accuracy matters "
            "more than sounding fluent. "
        )
    else:
        extra += "Write your 'reply' field in the same language the user used, unless they ask otherwise. "

    return instruction + "\n\n" + extra


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
        if st.session_state.study_subject in SUBJECT_KB_FILES:
            query_text = next((p for p in parts if isinstance(p, str)), None)
            if query_text:
                grounding_context = retrieve_subject_context(st.session_state.study_subject, query_text)
        instruction = build_study_instruction(st.session_state.study_subject, grounding_context)
    else:
        # .get() with a General fallback so no mode can ever KeyError here
        instruction = MODE_INSTRUCTIONS.get(st.session_state.mode, GENERAL_SYSTEM_INSTRUCTION)
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

for idx, msg in enumerate(st.session_state.history):
    avatar = EMOTIONS.get(msg.get("emotion", "neutral"), EMOTIONS["neutral"])["emoji"] if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_feedback_controls(idx, msg)

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
    # The key includes a counter that is bumped after each send, so the
    # uploader (and caption box) reset instead of keeping the old photo attached.
    photo = st.file_uploader(
        "Attach a photo",
        type=list(IMAGE_MIME_MAP.keys()),
        key=f"cheat_mind_photo_{st.session_state.photo_n}",
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
        key=f"cheat_mind_photo_caption_{st.session_state.photo_n}",
        placeholder="e.g. Can you help me with this question?",
    )
    if st.button("📤 Send photo", key="send_photo_btn"):
        ext = photo.name.rsplit(".", 1)[-1].lower()
        mime_type = IMAGE_MIME_MAP.get(ext, "image/jpeg")
        image_part = {"mime_type": mime_type, "data": photo.getvalue()}
        prompt_text = photo_caption.strip() or "Look at this photo and help me with it."
        fallback_text = f"[Photo attached] {photo_caption.strip()}" if photo_caption.strip() else "[Photo attached]"
        if api_key:
            st.session_state.photo_n += 1  # clears the uploader + caption on the next run
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