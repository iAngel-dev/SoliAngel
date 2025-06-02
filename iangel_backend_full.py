from flask import Flask, request, jsonify
import os
import json
import pickle

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("API_SECRET_KEY")

# Load models
with open(os.getenv("PHONE_SCAN_MODEL"), "rb") as f:
    phone_model = pickle.load(f)

with open(os.getenv("ANXIETY_MODEL"), "rb") as f:
    anxiety_model = pickle.load(f)

with open(os.getenv("FEATURE_SCALER"), "rb") as f:
    anxiety_scaler = pickle.load(f)

with open(os.getenv("USER_MODEL"), "rb") as f:
    user_model = pickle.load(f)

@app.route("/")
def home():
    return jsonify({"message": "Bienvenue sur l'API iAngel 🕊️"}), 200

@app.route("/scan-phone", methods=["POST"])
def scan_phone():
    data = request.get_json()
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "Aucune donnée transmise."}), 400
    prediction = phone_model.predict([features])
    return jsonify({"risque": int(prediction[0])})

@app.route("/analyze-anxiety", methods=["POST"])
def analyze_anxiety():
    data = request.get_json()
    features = data.get("features", [])
    if not features:
        return jsonify({"error": "Pas de données vocales fournies."}), 400
    scaled = anxiety_scaler.transform([features])
    result = anxiety_model.predict(scaled)
    return jsonify({"anxiété_probable": bool(result[0])})

@app.route("/voice", methods=["GET"])
def get_voice():
    return jsonify({"default_voice": os.getenv("DEFAULT_VOICE")})

@app.route("/teach", methods=["POST"])
def teach_context():
    topic = request.json.get("topic", "")
    try:
        with open("insf_knowledge_base.json", "r", encoding="utf-8") as f:
            base = json.load(f)
        return jsonify({"infos": base.get(topic.lower(), [])})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", 5000)))