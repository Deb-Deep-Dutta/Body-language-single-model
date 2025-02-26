# Author: Deb Deep Dutta
# Email: debdeepdutta42003@gmail.com
# Created: 2025-02-13
# Description: Flask backend for body language prediction with majority voting.
# License: MIT License
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd
import base64
import warnings
from flask import Flask, render_template, request, jsonify

# Initialize Flask app
app = Flask(__name__ , template_folder="templates")

# Setup mediapipe
mp_holistic = mp.solutions.holistic

# Load the pre-trained model
with open('body_language.pkl', 'rb') as f:
    model = pickle.load(f)

# Monkey-patch each DecisionTreeClassifier in the RandomForest (if present)
# This ensures that if the attribute 'monotonic_cst' is missing, it gets a default value.
def patch_tree(tree):
    if not hasattr(tree, 'monotonic_cst'):
        tree.monotonic_cst = None

# If your model is a Pipeline with a RandomForestClassifier at the end:
if hasattr(model, 'named_steps'):
    # Replace 'randomforestclassifier' with the actual step name if different.
    if 'randomforestclassifier' in model.named_steps:
        rf = model.named_steps['randomforestclassifier']
        for tree in rf.estimators_:
            patch_tree(tree)
else:
    # If your model is directly a RandomForestClassifier:
    if hasattr(model, 'estimators_'):
        for tree in model.estimators_:
            patch_tree(tree)

warnings.filterwarnings('ignore')

# Initialize mediapipe holistic model once for efficiency.
holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Remove header if present (e.g., "data:image/jpeg;base64,")
    img_data = data['image']
    if img_data.startswith("data:image"):
        _, img_data = img_data.split(',', 1)

    # Decode and convert the image
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Convert frame to RGB for mediapipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic_model.process(image_rgb)

    prediction = "No detection"
    try:
        if results.pose_landmarks and results.face_landmarks:
            pose_row = [val for landmark in results.pose_landmarks.landmark 
                        for val in [landmark.x, landmark.y, landmark.z, landmark.visibility]]
            face_row = [val for landmark in results.face_landmarks.landmark 
                        for val in [landmark.x, landmark.y, landmark.z, landmark.visibility]]
            row = pose_row + face_row

            X = pd.DataFrame([row])
            body_language_class = model.predict(X)[0]
            prediction = str(body_language_class)
    except Exception as e:
        # Log the error; the monkey-patch should prevent the 'monotonic_cst' error.
        print("Prediction error:", e)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
    # app.run(debug=True)
