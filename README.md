# Deep Learning Emotion Detection Web Application

An end-to-end Deep Learning project that classifies text into various emotional states (sadness, joy, love, anger, fear, and surprise) using sequential neural network architectures. The project compares **RNN**, **LSTM**, **GRU**, and **Bidirectional GRU (BiGRU)** models built with TensorFlow/Keras, and serves the best-performing model through a Streamlit web app and a FastAPI backend.

---

## 🚀 Features

* **Advanced Sequence Models**: Compares SimpleRNN, LSTM, GRU, and BiGRU performance metrics.
* **Deep Learning Pipeline**: Incorporates text tokenization, padding, class-weight balancing, and `EarlyStopping` optimization.
* **Streamlit UI**: Single-file interactive web app (`app.py`) for real-time inference.
* **FastAPI Backend**: Robust API endpoints for text preprocessing and multi-class emotion prediction with probability breakdowns.
* **Artifact Serialization**: Saves trained models (`.keras`) and tokenizers (`.pkl`) for seamless deployment.

---

## 🛠️ Tech Stack

* **Machine Learning / Deep Learning**: Python, TensorFlow, Keras, NumPy, Pandas
* **Web App**: Streamlit
* **Backend**: FastAPI, Uvicorn, Pydantic

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
│   ├── index.html                 # FastAPI frontend UI
│   ├── style.css                  # UI styling
│   └── script.js                  # Frontend logic & API fetch handling
│
├── app.py                         # Streamlit web application (single file)
├── Emotion_Detection.ipynb        # Colab notebook for EDA and model training
├── main.py                        # FastAPI application server
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

---

## ✅ Prerequisites

* **Python** 3.11 (the notebook was developed on Colab, so it is not required locally)
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

### Streamlit Web App (recommended)

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### FastAPI Backend

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

## ☁️ Deployment on Streamlit Community Cloud

1. Push this repository to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) and click **Create app**.
3. Select the repository and set the **Main file path** to `app.py`.
4. Click **Deploy**.

Streamlit Community Cloud will install the dependencies from `requirements.txt` and start the app automatically.

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

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ValueError: File not found: ... BiGRU_Model.keras` | The model file is missing or misnamed. Ensure `Artifacts/BiGRU_Model.keras` exists (exact spelling). |
| `Unrecognized keyword arguments passed to Embedding: {'quantization_config': None}` | The model was saved with a newer Keras than installed. Load/save it with a matching Keras version, or strip the incompatible fields from `config.json` inside the `.keras` archive. |
| `ModuleNotFoundError: No module named 'streamlit'` | Run `pip install -r requirements.txt` first, or `pip install streamlit`. |
| `[Errno 10048] error while attempting to bind on address ... 8000` | Port already in use. Kill the stale process or run on another port: `uvicorn main:app --port 8001`. |

---

## 📄 License

Distributed under the MIT License.