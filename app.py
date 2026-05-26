from flask import Flask, render_template, request, redirect, send_from_directory, url_for, jsonify
import numpy as np
import json
import uuid
import tensorflow as tf
import requests  # Added to communicate with Ollama
import gc        # Added for RAM management

app = Flask(__name__)

# Force TensorFlow to optimize CPU threads to save your i5 processor from freezing
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

model = tf.keras.models.load_model("D:/Projects/Plant-Disease-Recognition-System/models/plant_disease_recog_model_pwp.keras")

# Ollama Configuration (Using 1B model to fit in 8GB RAM)
# Ollama runs in Docker — port 11434 is mapped to host so localhost works fine
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:1b"
OLLAMA_TIMEOUT = 120  # seconds — cold-start on i5/8GB can take 60-90s the first time

label = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]

    # Force RAM cleanup after prediction finishes
    gc.collect()
    tf.keras.backend.clear_session()

    return prediction_label

@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
        full_path = f'{temp_name}_{image.filename}'
        image.save(full_path)

        prediction = model_predict(f'./{full_path}')

        # Pass the prediction to the UI so the chatbot route can read it later
        return render_template('home.html', result=True, imagepath=f'/{full_path}', prediction=prediction)
    else:
        return redirect('/')

# --- CHATBOT ROUTE ---
@app.route('/chat', methods=['POST'])
def chat_bot():
    data = request.get_json()
    user_message = data.get("message", "")
    current_disease = data.get("disease", "Unknown Plant Status")

    system_prompt = (
        f"You are an expert AI Botanist. The user's plant is diagnosed with: {current_disease}. "
        "Provide quick, accurate, short, and practical agricultural solutions to treat it. "
        "Keep your response friendly, clear, and under 4 sentences."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{system_prompt}\nUser Question: {user_message}\nAI Botanist Response:",
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        bot_response = response.json().get("response", "I encountered an issue generating a solution.")
        return jsonify({"response": bot_response.strip()})

    except requests.exceptions.ConnectionError:
        return jsonify({"response": "Cannot reach Ollama on localhost:11434. Check: docker ps"})
    except requests.exceptions.Timeout:
        return jsonify({"response": "Ollama timed out — the model is likely still loading into RAM. Wait 30 seconds and try again."})
    except requests.exceptions.HTTPError as e:
        return jsonify({"response": f"Ollama HTTP error: {e}. Try: docker exec ollama ollama pull llama3.2:1b"})
    except Exception as e:
        return jsonify({"response": f"Unexpected error: {str(e)}"})


# --- OLLAMA HEALTH CHECK ---
# Visit http://localhost:5000/ollama-status in your browser to debug connectivity
@app.route('/ollama-status', methods=['GET'])
def ollama_status():
    try:
        res = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m['name'] for m in res.json().get('models', [])]
        return jsonify({"status": "online", "models": models})
    except Exception as e:
        return jsonify({"status": "offline", "error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)