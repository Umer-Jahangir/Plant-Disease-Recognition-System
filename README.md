# 🌿 KISAAN — Plant Disease Recognition System

>AI-powered plant disease diagnosis with an integrated LLM chatbot

---

## What It Does

KISAAN lets you upload a photo of a plant leaf and get an instant disease diagnosis powered by a trained TensorFlow model. After diagnosis, an on-device AI Botanist (running via Ollama) answers your follow-up questions about treatment, prevention, and causes — all without sending your data to any external API.

---

## Features

- **Image-based disease detection** — Upload or drag-and-drop a leaf image; get a diagnosis in seconds
- **39 disease classes** across 14 crop types (Apple, Corn, Grape, Tomato, Potato, and more)
- **AI Botanist chatbot** — Diagnosis-aware LLM chat powered by `llama3.2:1b` via Ollama
- **Quick-prompt buttons** — One-click questions: organic treatments, contagion risk, seasonal prevention
- **Fully local inference** — No external API calls; everything runs on your machine
- **Low-resource optimized** — Tuned for an Intel i5 with 8 GB RAM

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · Flask |
| ML Model | TensorFlow / Keras (`.keras` format) |
| LLM | Ollama · `llama3.2:1b` (Docker) |
| Frontend | Jinja2 · Bootstrap 5 · Vanilla JS |
| Fonts | Cormorant Garamond · DM Sans |

---

## Supported Crops & Diseases

| Crop | Conditions Detected |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Pepper (Bell) | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Project Structure

```
Plant-Disease-Recognition-System/
│
├── app.py                        # Flask application & all routes
├── plant_disease.json            # Disease metadata (name, cause, cure, etc.)
│
├── models/
│   └── plant_disease_recog_model_pwp.keras   # Trained TF model (160×160 input)
│
├── templates/
│   └── home.html                 # Single-page Jinja2 UI
│
├── static/
│   ├── css/bootstrap.min.css
│   ├── js/bootstrap.bundle.min.js
│   └── images/logo.svg
│
└── uploadimages/                 # Temp directory for uploaded leaf images
```

---

## Prerequisites

- Python 3.9+
- Docker Desktop (for Ollama)
- ~3 GB disk space for the LLM model

---

## Downloading the Model

The trained Keras model is hosted on Hugging Face. Download it and place it inside the `models/` folder before running the app.

**Model:** [omarjahangeer/plant_disease_recognition](https://huggingface.co/omarjahangeer/plant_disease_recognition)
**License:** MIT

### Option A — Download manually (recommended for most users)

1. Go to [https://huggingface.co/omarjahangeer/plant_disease_recognition](https://huggingface.co/omarjahangeer/plant_disease_recognition)
2. Click **Files and versions**
3. Download `plant_disease_recog_model_pwp.keras`
4. Move the file into the `models/` folder:

```
Plant-Disease-Recognition-System/
└── models/
    └── plant_disease_recog_model_pwp.keras   ← place it here
```

### Option B — Download via Python

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="omarjahangeer/plant_disease_recognition",
    filename="plant_disease_recog_model_pwp.keras",
    local_dir="./models"
)
```

> Requires `pip install huggingface_hub`

### Option C — Load directly via Keras (no manual download)

```python
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
model = keras.saving.load_model("hf://omarjahangeer/plant_disease_recognition")
```

> If you use this approach, remove the `load_model` line in `app.py` and replace it with the above.

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Plant-Disease-Recognition-System.git
cd Plant-Disease-Recognition-System
```

### 2. Install Python dependencies

```bash
pip install flask tensorflow numpy requests huggingface_hub
```

### 3. Download the model

Follow the [Downloading the Model](#downloading-the-model) section above and place the `.keras` file in `models/`.

### 4. Start Ollama in Docker

```bash
docker run -d -p 11434:11434 --name ollama ollama/ollama
docker exec ollama ollama pull llama3.2:1b
```

### 5. Update the model path in `app.py`

```python
# Line 13 — use a relative path so it works on any machine
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
```

### 6. Run the Flask app

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Usage

1. Open the app in your browser
2. Drag-and-drop or click to upload a leaf image (JPG, PNG, or WEBP)
3. Click **Analyze Plant** and wait for the model to process
4. View the diagnosis — disease name, cause, and recommended cure
5. Use the **AI Botanist** chatbot to ask follow-up questions

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Home page |
| `POST` | `/upload/` | Upload image and return prediction |
| `POST` | `/chat` | Send a message to the AI Botanist |
| `GET` | `/ollama-status` | Check if Ollama is online and list loaded models |
| `GET` | `/uploadimages/<filename>` | Serve uploaded images |

### `/chat` — Request body

```json
{
  "message": "What organic treatments are available?",
  "disease": "Tomato___Early_blight"
}
```

---

## Debugging

**Ollama not reachable?**
```bash
docker ps                                        # confirm container is running
curl http://localhost:11434/api/tags             # check available models
# or visit http://localhost:5000/ollama-status
```

**Model response timing out?**
The first inference on an i5/8 GB machine can take 60–90 seconds as the model loads into RAM. The timeout is set to 120 seconds. Subsequent queries are faster.

**TF using too much CPU?**
Thread limits are hardcoded in `app.py` to keep your system responsive:
```python
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)
```
Raise these values if you have more cores available.

---

## Image Requirements & Best Practices

The model was trained on clean, isolated leaf images. The quality of your photo directly impacts diagnosis accuracy.

**For best results:**
- Use a **plain background** — white paper, a clean table, or a solid-color surface works well
- Photograph **one leaf at a time**; avoid bunching multiple leaves together
- Fill the frame with the leaf — get close enough that the leaf takes up most of the image
- Shoot in **natural daylight** or good indoor light; avoid harsh shadows or flash glare
- Keep the camera steady; blurry images reduce accuracy significantly
- Accepted formats: **JPG, PNG, WEBP**

**Avoid:**
- Soil, grass, or sky in the background
- Images taken from far away where the leaf is small
- Multiple overlapping leaves in one shot
- Heavily filtered or edited photos

> The model input is resized to **160×160 px** internally. This means fine texture details can get lost in low-quality or distant shots — another reason to photograph the leaf up close.

---

## Known Limitations

- `llama3.2:1b` is a small model — answers are practical but not exhaustive. Do not rely solely on its output for crop management decisions.
- Uploaded images are saved to `/uploadimages/` with UUIDs but are **not automatically deleted**. Add a cleanup routine for production use.
- The app runs in Flask's built-in development server. Use Gunicorn or uWSGI before deploying anywhere public.

---

## License

You can contribute it will be appreciatable.