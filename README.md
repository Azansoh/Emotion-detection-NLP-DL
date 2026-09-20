---
title: Emotion Detection Api
emoji: 💻
colorFrom: pink
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Deep Learning Emotion Detection Web Application

An end-to-end Deep Learning project that classifies text into various emotional states (sadness, joy, love, anger, fear, and surprise) using sequential neural network architectures. The project compares **RNN**, **LSTM**, **GRU**, and **Bidirectional GRU (BiGRU)** models built with TensorFlow/Keras, and serves the best-performing model through a modern web UI powered by FastAPI.

---

## 🚀 Features

* **Advanced Sequence Models**: Compares SimpleRNN, LSTM, GRU, and BiGRU performance metrics.
* **Deep Learning Pipeline**: Incorporates text tokenization, padding, class-weight balancing, and `EarlyStopping` optimization.
* **FastAPI Backend**: Robust API endpoints for text preprocessing and multi-class emotion prediction with probability breakdowns.
* **Interactive Frontend**: A custom, clean web user interface (`index.html`, `style.css`, `script.js`) for real-time inference.
* **Artifact Serialization**: Saves trained models (`.keras`) and tokenizers (`.pkl`) for seamless deployment.

---

## 🛠️ Tech Stack

* **Machine Learning / Deep Learning**: Python, TensorFlow, Keras, Scikit-Learn, NumPy, Pandas
* **Backend**: FastAPI, Uvicorn, Pydantic
* **Frontend**: HTML5, CSS3, JavaScript

---

## 📁 Project Structure

```text
📦 DL(NLP) Project UI
│
├── 📂 Artifacts
│   ├── BiGRU_Model.keras          # Trained Bidirectional GRU model
│   └── tokenizer.pkl              # Fitted text tokenizer
│
├── 📂 static
│   ├── index.html                 # Frontend user interface
│   ├── style.css                  # UI styling
│   └── script.js                  # Frontend logic & API fetch handling
│
├── Emotion_Detection.ipynb        # Colab notebook for EDA and model training
├── main.py                        # FastAPI application server
├── requirements.txt               # Python package dependencies
├── runtime.txt                    # Deployment runtime configuration
└── README.md                      # Project documentation
```

---

## ✅ Prerequisites

* **Python** 3.9 – 3.11 (the notebook was developed on Colab, so it is not required locally)
* **pip** (comes bundled with Python)

---

## 📦 Installation

```bash
# 1. Clone the repository (or open the project folder)
git clone <your-repo-url>
cd "DL(NLP) Project UI"

# 2. (Recommended) Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **Note:** The trained artifacts (`Artifacts/BiGRU_Model.keras` and `Artifacts/tokenizer.pkl`) must be present. If not, run `Emotion_Detection.ipynb` to train the models, then place the saved files inside the `Artifacts/` folder.

---

## 🚀 Running the Application

```bash
uvicorn main:app --reload
```

Then open **http://127.0.0.1:8000** in your browser.

| Endpoint   | Method | Description |
|------------|--------|-------------|
| `/`        | GET    | Serves the web UI |
| `/health`  | GET    | Health check — confirms the model is loaded |
| `/predict` | POST   | Predicts the emotion for input text |

---

## 🔗 API Usage

### Health Check

```bash
curl http://127.0.0.1:8000/health
```

**Response:**
```json
{"status": "Server is running", "model_loaded": true}
```

### Predict Emotion

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I feel so happy and excited"}'
```

**Response:**
```json
{
  "text": "I feel so happy and excited",
  "predicted_emotion": "joy",
  "confidence": 0.99769,
  "all_probabilites": {
    "sadness": 0.0002,
    "joy": 0.9977,
    "love": 0.0009,
    "anger": 0.0005,
    "fear": 0.0001,
    "surprise": 0.0006
  }
}
```

---

## ☁️ Deployment

The project is ready for platform-as-a-service deployment (e.g. **Render**, **Heroku**, **Railway**):

* `runtime.txt` specifies the Python version (`python-3.11.9`).
* `requirements.txt` holds all dependencies.
* Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

---

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ValueError: File not found: ... BiGRU_Model.keras` | The model file is missing or misnamed. Ensure `Artifacts/BiGRU_Model.keras` exists (exact spelling). |
| `Unrecognized keyword arguments passed to Embedding: {'quantization_config': None}` | The model was saved with a newer Keras than installed. Load/save it with a matching Keras version, or strip the incompatible fields from `config.json` inside the `.keras` archive. |
| `[Errno 10048] error while attempting to bind on address ... 8000` | Port already in use. Kill the stale process or run on another port: `uvicorn main:app --port 8001`. |
| `model_loaded: false` at `/health` | The tokenizer or model failed to load at startup — check the server console logs. |

---

## 📄 License

Distributed under the MIT License.
