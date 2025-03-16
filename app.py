from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import json
import os

app = Flask(__name__)

# Define paths—adjust these paths as needed for your local setup
MODEL_PATH = "audio_recognition_model.h5"  # Local path to your saved model
LABELS_FILE = "label_mapping.json"          # Local path to your label mapping file

# Load the model
loaded_model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully from", MODEL_PATH)

# Load label mapping (expects a JSON file, e.g., {"0": "Label A", "1": "Label B", ...})
try:
    with open(LABELS_FILE, 'r') as f:
        labels = json.load(f)
    print("Loaded labels:", labels)
except Exception as e:
    print("Label file not found or could not be loaded. Using model output as is.")
    labels = None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file was provided
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Please upload a WAV file with the key 'file'."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected. Please upload a valid WAV file."}), 400

    # Validate file extension
    if not file.filename.lower().endswith('.wav'):
        return jsonify({"error": "Invalid file type. Please upload a .wav file."}), 400

    # Save the file temporarily
    temp_filename = "temp.wav"
    file.save(temp_filename)

    # Load and preprocess the audio file
    try:
        audio, sr = librosa.load(temp_filename, sr=None)
    except Exception as e:
        os.remove(temp_filename)
        return jsonify({"error": "Error processing audio file.", "details": str(e)}), 500

    # Remove the temporary file
    os.remove(temp_filename)

    # Compute MFCC features
    n_mfcc = 39
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc = mfcc.T  # Transpose to shape (time, n_mfcc)

    # Ensure the MFCC array has a fixed number of frames (e.g., 719)
    target_frames = 719
    num_frames = mfcc.shape[0]
    if num_frames < target_frames:
        pad_width = target_frames - num_frames
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    elif num_frames > target_frames:
        mfcc = mfcc[:target_frames, :]

    # Add batch dimension to match model input shape: (1, 719, 39)
    mfcc_input = np.expand_dims(mfcc, axis=0)

    # Get prediction from the model
    predictions = loaded_model.predict(mfcc_input)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions))

    # Map predicted index to a human-readable label if available
    if labels is not None:
        if isinstance(labels, list):
            predicted_label = labels[predicted_index]
        elif isinstance(labels, dict):
            predicted_label = labels.get(str(predicted_index), str(predicted_index))
        else:
            predicted_label = str(predicted_index)
    else:
        predicted_label = str(predicted_index)

    return jsonify({
        "predicted_label": predicted_label,
        "confidence": confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
