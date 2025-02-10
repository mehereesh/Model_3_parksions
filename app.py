from flask import Flask, render_template, request, jsonify, send_file
import librosa
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import uuid
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model and scaler
with open('parkinsons_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Create a folder for temporary file storage
os.makedirs("temp", exist_ok=True)

# Feature extraction function
def extract_features(audio_file):
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(audio_file, sr=None)
        print(f"Audio file loaded successfully with sample rate {sr}.")

        # Extract features
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr).T, axis=0)

        # Combine features into a single vector (Ensure it matches training features)
        features = np.hstack([mfccs, chroma, zcr, rms, spectral_centroid, spectral_bandwidth, spectral_rolloff])[:22]
        print(f"Extracted {len(features)} features from the audio.")
        return features
    except Exception as e:
        print(f"Error extracting features from audio: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'result': 'No audio file uploaded'})

    # Save the uploaded audio file with a unique name
    audio_file = request.files['audio']
    unique_filename = f"{uuid.uuid4()}_{audio_file.filename}"
    file_path = os.path.join("temp", unique_filename)
    audio_file.save(file_path)

    # Check if the file exists
    if not os.path.exists(file_path):
        return jsonify({'result': 'File upload failed'})

    print(f"File uploaded: {file_path}")

    # Extract features from the audio
    features = extract_features(file_path)

    # Clean up the temporary file
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File {file_path} not found for deletion")

    if features is None:
        return jsonify({'result': 'Error extracting features from audio'})

    # Standardize using the same scaler from training
    features_scaled = scaler.transform(features.reshape(1, -1))

    # Predict using the model
    prediction = model.predict(features_scaled)
    prediction_proba = model.predict_proba(features_scaled)[0]

    # Generate graph for accuracy
    labels = ["Healthy", "Parkinson's"]
    plt.bar(labels, prediction_proba, color=['green', 'red'])
    plt.title('Prediction Confidence')
    plt.ylabel('Probability')
    plt.ylim(0, 1)

    # Generate a unique filename for the graph
    graph_filename = f"confidence_{uuid.uuid4().hex}.png"
    confidence_graph_path = os.path.join('temp', graph_filename)
    plt.savefig(confidence_graph_path)
    plt.close()

    # Prepare result message
    result = "Parkinson's Detected" if prediction[0] == 1 else "Healthy"
    accuracy = prediction_proba[1] if prediction[0] == 1 else prediction_proba[0]

    return jsonify({'result': result, 'graph': f'/graph/{graph_filename}', 'accuracy': accuracy})

@app.route('/graph/<filename>')
def graph(filename):
    graph_path = os.path.join('temp', filename)
    if os.path.exists(graph_path):
        return send_file(graph_path, mimetype='image/png')
    return jsonify({'result': 'Graph not found'})

if __name__ == '__main__':
    app.run(debug=True)
