"""
Moodline — read the emotion in your words.

Single-file Streamlit app for the BiGRU emotion classifier.
Runs with:  streamlit run app.py
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import html
import json
import re
from pathlib import Path

import numpy as np
import streamlit as st
from keras.models import load_model

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Artifacts" / "BiGRU_Model.keras"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer.json"

MAX_SEQUENCE_LENGTH = 50

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

EMOTION_COLORS = {
    "sadness": "#5b7fde",
    "joy": "#f2b705",
    "love": "#e85d75",
    "anger": "#e4572e",
    "fear": "#8b5fbf",
    "surprise": "#17bebb",
}


# ------------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def texts_to_sequences(data, text: str) -> list[int]:
    word_index = data["word_index"]
    num_words = data["num_words"]
    oov_index = word_index.get(data["oov_token"])

    seq: list[int] = []
    for word in text.split():
        idx = word_index.get(word)
        if idx is not None and idx < num_words:
            seq.append(idx)
        elif oov_index is not None:
            seq.append(oov_index)
    return seq


def pad_post(seq: list[int], maxlen: int = MAX_SEQUENCE_LENGTH) -> np.ndarray:
    arr = np.zeros((1, maxlen), dtype=np.int64)
    n = min(len(seq), maxlen)
    arr[0, :n] = seq[:n]
    return arr


@st.cache_resource(show_spinner="Waking the model up…")
def load_bi_gru_model():
    return load_model(MODEL_PATH)


def predict(model, tokenizer_data, text: str) -> tuple[str, float, dict[str, float]]:
    cleaned = preprocess_text(text)
    padded = pad_post(texts_to_sequences(tokenizer_data, cleaned))

    probabilities = model.predict(padded, verbose=0)[0]

    top_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) for prob, label in zip(probabilities, EMOTION_LABELS)
    }

    return (
        EMOTION_LABELS[top_index],
        float(probabilities[top_index]),
        all_probabilities,
    )


# ------------------------------------------------------------------
# Theming (presentation only)
# ------------------------------------------------------------------
def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def inject_theme():
    st.markdown(
        """<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500;1,9..144,600&family=Instrument+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #0f0e13;
    --panel: #17161d;
    --panel-2: #1e1c26;
    --line: rgba(255, 244, 230, 0.10);
    --line-strong: rgba(255, 244, 230, 0.22);
    --text: #f6f1e9;
    --text-soft: #c9c3d3;
    --text-mute: #9c96a9;
    --display: "Fraunces", Georgia, "Times New Roman", serif;
    --body: "Instrument Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* ---------- page ---------- */
.stApp {
    background:
        radial-gradient(900px 520px at 88% -8%, rgba(139, 95, 191, 0.20), transparent 60%),
        radial-gradient(760px 480px at -6% 4%, rgba(23, 190, 187, 0.13), transparent 60%),
        radial-gradient(700px 500px at 50% 112%, rgba(232, 93, 117, 0.10), transparent 60%),
        var(--ink);
    color: var(--text);
    font-family: var(--body);
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container {
    max-width: 720px;
    padding-top: 2.6rem;
    padding-bottom: 4rem;
}

/* ---------- masthead ---------- */
.masthead {
    position: relative;
    overflow: hidden;
    background: linear-gradient(160deg, var(--panel) 0%, var(--panel-2) 100%);
    border: 1px solid var(--line);
    border-radius: 28px;
    padding: 2.1rem 2.3rem 1.9rem;
    margin-bottom: 2.2rem;
    box-shadow: 0 30px 60px -30px rgba(0, 0, 0, 0.8), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
.masthead::after {
    content: "";
    position: absolute;
    right: -70px; top: -70px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: conic-gradient(from 200deg,
        #5b7fde, #17bebb, #f2b705, #e4572e, #e85d75, #8b5fbf, #5b7fde);
    filter: blur(58px);
    opacity: 0.32;
    pointer-events: none;
}
.mood-title {
    position: relative;
    z-index: 1;
    font-family: var(--display);
    font-weight: 600;
    font-size: 2.5rem;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin: 0;
    color: var(--text);
}
.mood-title span { color: #f2b705; margin-right: 0.35rem; font-size: 0.8em; }
.mood-tag {
    position: relative;
    z-index: 1;
    color: var(--text-soft);
    font-size: 1rem;
    line-height: 1.55;
    max-width: 30rem;
    margin: 0.7rem 0 0 0;
}
.palette-row {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1.3rem;
}
.palette-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.28rem 0.7rem 0.28rem 0.55rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.03);
    color: var(--text-soft);
    font-size: 0.82rem;
    font-weight: 500;
}
.palette-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ---------- input ---------- */
div[data-testid="stTextArea"] label p {
    font-family: var(--display);
    font-size: 1.15rem !important;
    letter-spacing: -0.01em;
    color: var(--text) !important;
}
div[data-testid="stTextArea"] div[data-baseweb="textarea"],
div[data-testid="stTextArea"] div[data-baseweb="base-input"] {
    background: var(--panel) !important;
    border-radius: 18px !important;
    border: 1px solid var(--line-strong) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
div[data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
    border-color: rgba(246, 241, 233, 0.55) !important;
    box-shadow: 0 0 0 4px rgba(246, 241, 233, 0.08) !important;
}
div[data-testid="stTextArea"] textarea {
    background: transparent !important;
    color: #ffffff !important;
    font-family: var(--body);
    font-size: 1.05rem;
    line-height: 1.65;
    padding: 1.05rem 1.15rem;
    caret-color: #f2b705;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: var(--text-mute) !important;
    opacity: 1 !important;
}

/* ---------- button ---------- */
.stButton > button {
    font-family: var(--body);
    border-radius: 999px;
    padding: 0.7rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    background: var(--text);
    color: var(--ink);
    border: none;
    box-shadow: 0 8px 22px -8px rgba(246, 241, 233, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
}
.stButton > button:hover {
    background: #ffffff;
    color: #000000;
    transform: translateY(-1px);
    box-shadow: 0 12px 28px -8px rgba(255, 255, 255, 0.4);
}
.stButton > button:active { transform: translateY(0); }
.stButton > button:focus-visible {
    outline: 2px solid #f2b705;
    outline-offset: 3px;
}
.stButton > button:disabled {
    background: rgba(246, 241, 233, 0.18);
    color: var(--text-mute);
    box-shadow: none;
}

/* ---------- result card ---------- */
.emotion-card {
    position: relative;
    overflow: hidden;
    margin-top: 2rem;
    padding: 2.4rem 2.4rem 2rem;
    text-align: center;
    border-radius: 30px;
    border: 1px solid var(--accent-line);
    background:
        radial-gradient(520px 260px at 50% 0%, var(--accent-soft), transparent 70%),
        linear-gradient(180deg, var(--panel), var(--panel-2));
    box-shadow: 0 34px 70px -34px var(--accent-glow), inset 0 1px 0 rgba(255, 255, 255, 0.05);
    animation: reveal 0.7s cubic-bezier(0.2, 0.8, 0.2, 1) both;
}
.confidence-ring {
    width: 150px; height: 150px;
    margin: 0 auto 1.1rem;
    padding: 8px;
    border-radius: 50%;
    box-shadow: 0 0 40px -6px var(--accent-glow);
}
.confidence-ring-inner {
    width: 100%; height: 100%;
    border-radius: 50%;
    background: var(--panel);
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: inset 0 0 0 1px var(--line);
}
.emotion-emoji { font-size: 3.6rem; line-height: 1; }
.emotion-word {
    font-family: var(--display);
    font-style: italic;
    font-weight: 600;
    font-size: 3.4rem;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin: 0.2rem 0 0.35rem;
}
.emotion-conf {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid var(--line);
    color: #ffffff !important;
    font-weight: 600;
    font-size: 0.92rem;
    font-variant-numeric: tabular-nums;
}
.echoed {
    margin: 1.5rem auto 0;
    padding-top: 1.2rem;
    max-width: 34rem;
    border-top: 1px solid var(--line);
    color: var(--text-soft);
    font-family: var(--display);
    font-style: italic;
    font-size: 1.12rem;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

/* ---------- breakdown ---------- */
.breakdown-wrap {
    margin-top: 1.6rem;
    padding: 1.6rem 1.8rem 1.1rem;
    border-radius: 24px;
    border: 1px solid var(--line);
    background: var(--panel);
}
.breakdown-header {
    font-family: var(--display);
    color: #ffffff;
    font-size: 1.3rem;
    letter-spacing: -0.01em;
    margin: 0 0 1.1rem 0;
}
.bar-row {
    display: flex;
    align-items: center;
    gap: 0.9rem;
    margin-bottom: 0.9rem;
    font-size: 0.93rem;
}
.bar-label {
    min-width: 7.6rem;
    color: var(--text-soft);
    font-weight: 500;
}
.bar-label.top { color: #ffffff; font-weight: 700; }
.bar-track {
    flex: 1;
    height: 11px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 999px;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    border-radius: 999px;
    animation: grow 0.9s cubic-bezier(0.2, 0.8, 0.2, 1) both;
    animation-delay: 0.25s;
}
.bar-value {
    min-width: 3.9rem;
    text-align: right;
    color: var(--text-soft);
    font-weight: 600;
    font-variant-numeric: tabular-nums;
}
.bar-value.top { color: #ffffff; font-weight: 700; }

/* ---------- motion ---------- */
@keyframes reveal {
    from { opacity: 0; transform: translateY(16px) scale(0.985); }
    to   { opacity: 1; transform: none; }
}
@keyframes grow { from { width: 0; } }
@media (prefers-reduced-motion: reduce) {
    .emotion-card, .bar-fill { animation: none !important; }
    .stButton > button { transition: none; }
}

/* ---------- small screens ---------- */
@media (max-width: 600px) {
    .masthead { padding: 1.6rem 1.4rem; border-radius: 22px; }
    .mood-title { font-size: 2rem; }
    .emotion-card { padding: 1.8rem 1.2rem 1.5rem; }
    .emotion-word { font-size: 2.6rem; }
    .bar-label { min-width: 6.2rem; }
    .breakdown-wrap { padding: 1.3rem 1.1rem 0.8rem; }
}
</style>""",
        unsafe_allow_html=True,
    )


def apply_mood_glow(color: str) -> None:
    """Tint the page background with the detected emotion's colour."""
    st.markdown(
        "<style>.stApp{background:"
        f"radial-gradient(900px 560px at 50% 38%, {hex_to_rgba(color, 0.16)}, transparent 65%),"
        "radial-gradient(900px 520px at 88% -8%, rgba(139,95,191,0.14), transparent 60%),"
        "radial-gradient(760px 480px at -6% 4%, rgba(23,190,187,0.10), transparent 60%),"
        "#0f0e13;}</style>",
        unsafe_allow_html=True,
    )


def render_masthead() -> None:
    chips = "".join(
        f'<span class="palette-chip">'
        f'<span class="palette-dot" style="background:{EMOTION_COLORS[label]};"></span>'
        f"{label.title()}</span>"
        for label in EMOTION_LABELS
    )
    st.markdown(
        '<div class="masthead">'
        '<p class="mood-title"><span>✦</span>Moodline</p>'
        '<p class="mood-tag">Deep Learning NLP Instrument That Reads the Hidden Emotion Inside Your Words</p>'
        f'<div class="palette-row">{chips}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def render_result(emotion: str, confidence: float, sentence: str, color: str) -> None:
    pct = confidence * 100
    safe_sentence = html.escape(sentence)
    st.markdown(
        f'<div class="emotion-card" style="'
        f"--accent:{color};"
        f"--accent-soft:{hex_to_rgba(color, 0.20)};"
        f"--accent-glow:{hex_to_rgba(color, 0.55)};"
        f'--accent-line:{hex_to_rgba(color, 0.40)};">'
        f'<div class="confidence-ring" style="background:conic-gradient({color} {pct:.1f}%, rgba(255,255,255,0.08) 0);">'
        f'<div class="confidence-ring-inner">'
        f'<div class="emotion-emoji">{EMOTION_EMOJIS.get(emotion, "🙂")}</div>'
        f"</div></div>"
        f'<div class="emotion-word" style="color:{color};">{emotion.title()}</div>'
        f'<div class="emotion-conf">{pct:.1f}% confidence</div>'
        f'<div class="echoed">“{safe_sentence}”</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_breakdown(probabilities: dict[str, float], color: str) -> None:
    rows = []
    ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for rank, (label, value) in enumerate(ordered):
        top = " top" if rank == 0 else ""
        bar_color = EMOTION_COLORS.get(label, color)
        glow = f"box-shadow:0 0 10px {hex_to_rgba(bar_color, 0.6)};" if rank == 0 else ""
        rows.append(
            f'<div class="bar-row">'
            f'<span class="bar-label{top}">{EMOTION_EMOJIS[label]} {label.title()}</span>'
            f'<div class="bar-track"><div class="bar-fill" '
            f'style="width:{value * 100:.1f}%;background:{bar_color};{glow}"></div></div>'
            f'<span class="bar-value{top}">{value * 100:.1f}%</span>'
            f"</div>"
        )
    st.markdown(
        '<div class="breakdown-wrap">'
        '<p class="breakdown-header">Confidence Breakdown</p>'
        + "".join(rows)
        + "</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# App Layout
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Moodline — emotion detector",
    page_icon="✎",
    layout="centered",
)

inject_theme()
render_masthead()

with st.spinner("Waking the model up…"):
    try:
        tokenizer_data = load_tokenizer()
        model = load_bi_gru_model()
        model_ready = True
    except Exception as exc:
        model_ready = False
        st.error(f"Could not load the model: {exc}")

text = st.text_area(
    "**Enter your sentence below:**",
    placeholder="e.g. “I can't believe we actually pulled this off.”",
    height=130,
    max_chars=2000,
)

analyze = st.button("Read the mood", type="primary", disabled=not model_ready)

if analyze:
    sentence = text.strip()
    if not sentence:
        st.warning("Write a sentence first, then read its mood.")
    else:
        try:
            emotion, confidence, probabilities = predict(model, tokenizer_data, sentence)
            color = EMOTION_COLORS.get(emotion, "#6c7a89")

            apply_mood_glow(color)
            render_result(emotion, confidence, sentence, color)
            render_breakdown(probabilities, color)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")