from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import io
from PIL import Image

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_fold0_55.tflite")
prediction_fn = interpreter.get_signature_runner("serving_default")

# MediaPipe initialization
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Sign language labels
SIGN_LABELS = {
    0: 'here', 1: 'there', 2: 'go', 3: 'time', 4: 'correct', 
    # ... (rest of the labels)
}

SEQUENCE_LENGTH = 30
current_sequence = []

def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh, rh])

def get_top3_predictions(prediction):
    outputs = prediction['outputs']
    top3_idx = np.argsort(outputs)[-3:][::-1]
    probs = np.exp(outputs[top3_idx]) / np.sum(np.exp(outputs))
    
    predictions = []
    for idx, prob in zip(top3_idx, probs):
        predictions.append({
            'word': SIGN_LABELS[idx],
            'probability': float(prob)
        })
    return predictions

def process_frame(frame_data):
    global current_sequence

    # Convert base64 to image
    _, encoded = frame_data.split(",", 1)
    frame = np.array(Image.open(io.BytesIO(base64.b64decode(encoded))))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process with MediaPipe
    frame.flags.writeable = False
    results = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    frame.flags.writeable = True

    # Extract keypoints if hands are detected
    if results.left_hand_landmarks or results.right_hand_landmarks:
        keypoints = extract_keypoints(results)
        current_sequence.append(keypoints)
        current_sequence = current_sequence[-SEQUENCE_LENGTH:]
        
        # Make prediction if we have enough frames
        if len(current_sequence) == SEQUENCE_LENGTH:
            sequence_np = np.array(current_sequence, dtype=np.float32)
            prediction = prediction_fn(inputs=sequence_np)
            predictions = get_top3_predictions(prediction)
            
            # Check if top prediction is confident enough
            if predictions[0]['probability'] > 0.7:  # Confidence threshold
                return predictions, predictions[0]['word']
            return predictions, None
            
    return None, None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(frame_data):
    predictions, final_word = process_frame(frame_data)
    if predictions:
        emit('predictions', {
            'predictions': predictions,
            'final_word': final_word
        })

@socketio.on('delete_word')
def handle_delete():
    global current_sequence
    current_sequence = []

if __name__ == '__main__':
    socketio.run(app, debug=True)