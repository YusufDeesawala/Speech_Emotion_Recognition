from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa  # for audio processing
import tensorflow as tf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('models/model.h5')

emotions = ['disgust', 'fear', 'neutral', 'happy', 'sad', 'ps', 'angry']

# Upload folder for the audio file
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Helper function to process audio file and extract features
def extract_audio_features(audio_path, num_mfcc=40):  # Use the same number of MFCCs as in training
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    return np.mean(mfcc.T, axis=0) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file securely
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Extract audio features
    features = extract_audio_features(file_path)

    # Reshape the feature array to match the input shape of the model (e.g., (1, n_features))
    features = features.reshape(1, -1)

    # Make prediction using the model
    prediction = model.predict(features)

    # Get the predicted class (highest probability)
    predicted_idx = np.argmax(prediction, axis=1).item()

    # Map index to emotion label
    predicted_emotion = emotions[predicted_idx]

    # Return the result as a response
    return jsonify({'emotion': predicted_emotion}), 200

if __name__ == '__main__':
    app.run(debug=True)
