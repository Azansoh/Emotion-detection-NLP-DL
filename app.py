"""
Moodline — read the emotion in your words.

Single-file Streamlit app for the BiGRU emotion classifier.
Runs with:  streamlit run app.py

Uses the Keras model on the PyTorch backend (no TensorFlow), so it
installs and runs on any recent Python, including 3.14.
"""

import os

# Must be set before importing keras (no TensorFlow on Python 3.14).
os.environ.setdefault("KERAS_BACKEND", "torch")

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
# Text preprocessing (must match training-time format)
# ------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------------------
# Tokenizer (JSON snapshot of the training-time Keras legacy Tokenizer)
# ------------------------------------------------------------------
@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def texts_to_sequences(data, text: str) -> list[int]:
    """Replicates the Keras legacy Tokenizer (num_words + oov)."""
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


# ------------------------------------------------------------------
# Model loading (cached — loads once per session)
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Waking the model up…")
def load_bi_gru_model():
    return load_model(MODEL_PATH)


# ------------------------------------------------------------------
# Prediction
# ------------------------------------------------------------------
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
# Theming
# ------------------------------------------------------------------
def inject_theme():
    st.markdown(
        """
        <style>
        .stApp {
            background: #0e0f14;
            color: #f2f0ea;
        }
        .mood-title {
            font-family: "Fraunces", Georgia, serif;
            font-weight: 600;
            font-size: 1.4rem;
            letter-spacing: -0.01em;
            margin: 0;
        }
        .mood-tag {
            color: #9497a6;
            font-size: 0.82rem;
            margin: 0.15rem 0 2rem;
        }
        .emotion-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 2rem 2.5rem;
            background: linear-gradient(180deg, #16171f, #1c1e29);
            text-align: center;
            margin-top: 1.2rem;
        }
        .emotion-emoji { font-size: 3rem; line-height: 1; }
        .emotion-word {
            font-family: "Fraunces", Georgia, serif;
            font-style: italic;
            font-weight: 600;
            font-size: 3rem;
            margin: 0.25rem 0 0.1rem;
        }
        .emotion-conf {
            font-family: monospace;
            color: #9497a6;
            font-size: 0.85rem;
        }
        .echoed {
            margin-top: 1.2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            color: #9497a6;
            font-style: italic;
            font-size: 0.95rem;
        }
        div[data-testid="stTextArea"] textarea {
            background: rgba(0,0,0,0.18);
            color: #f2f0ea;
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            font-size: 1rem;
            line-height: 1.55;
        }
        div[data-testid="stTextArea"] textarea:focus {
            border-color: #6c7a89;
        }
        .stButton > button {
            border-radius: 999px;
            padding: 0.65rem 1.6rem;
            font-weight: 600;
            background: #f2f0ea;
            color: #0e0f14;
            border: none;
        }
        .stButton > button:hover {
            background: #6c7a89;
            color: #fff;
            border: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_breakdown(probabilities: dict[str, float], color: str) -> None:
    st.markdown("#### Breakdown")
    for label, value in sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True):
        bar_html = (
            f'<div style="display:flex;align-items:center;gap:0.75rem;'
            f'margin-bottom:0.6rem;font-size:0.85rem;">'
            f'<span style="min-width:7rem;color:#9497a6;">{EMOTION_EMOJIS[label]} {label.title()}</span>'
            f'<div style="flex:1;height:8px;background:rgba(255,255,255,0.06);'
            f'border-radius:999px;overflow:hidden;">'
            f'<div style="width:{value * 100:.1f}%;height:100%;background:{color};'
            f'border-radius:999px;"></div>'
            f'</div>'
            f'<span style="min-width:3.2rem;text-align:right;font-family:monospace;'
            f'color:#5c5f6d;">{value * 100:.1f}%</span>'
            f'</div>'
        )
        st.markdown(bar_html, unsafe_allow_html=True)


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Moodline — emotion detector",
    page_icon="✎",
    layout="centered",
)

inject_theme()

st.markdown('<p class="mood-title">✦ Moodline</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="mood-tag">an instrument that reads the mood inside a sentence</p>',
    unsafe_allow_html=True,
)

with st.spinner("Waking the model up…"):
    try:
        tokenizer_data = load_tokenizer()
        model = load_bi_gru_model()
        model_ready = True
    except Exception as exc:  # pragma: no cover - surfaced to the user
        model_ready = False
        st.error(f"Could not load the model: {exc}")

text = st.text_area(
    "**Type a sentence and let it speak for itself — e.g. “I can't believe we actually pulled this off.”**",
    placeholder="Write a sentence and let it speak for itself — e.g. “I can't believe we actually pulled this off.”",
    height=140,
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

            st.markdown(
                f"""
                <div class="emotion-card" style="box-shadow: 0 20px 40px -18px {color};">
                    <div class="emotion-emoji">{EMOTION_EMOJIS.get(emotion, "🙂")}</div>
                    <div class="emotion-word" style="color:{color};">{emotion.title()}</div>
                    <div class="emotion-conf">{confidence * 100:.1f}% confidence</div>
                    <div class="echoed">“{sentence}”</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("---")
            render_breakdown(probabilities, color)
        except Exception as exc:  # pragma: no cover - surfaced to the user
            st.error(f"Prediction failed: {exc}")