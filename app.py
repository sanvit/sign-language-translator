from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import io
from PIL import Image
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_fold0_55.tflite")
prediction_fn = interpreter.get_signature_runner("serving_default")

# MediaPipe initialization - moved to session handling
mp_holistic = mp.solutions.holistic

# Sign language labels
ORD2SIGN2 = {0: 'here', 1: 'there', 2: 'go', 3: 'time', 4: 'correct', 5: 'taxi', 6: 'money',
             7: 'confirm', 8: 'become', 9: 'card', 10: 'subway', 11: 'Myeongdong', 12: 'Songpa',
             13: 'train', 14: 'arrive', 15: 'place', 16: 'next', 17: 'get off', 18: 'what',
             19: 'not work', 20: 'destination', 21: 'bus', 22: 'none', 23: 'see', 24: 'police',
             25: 'school', 26: 'intersection', 27: 'cross', 28: 'traffic light', 29: 'right turn',
             30: 'before', 31: 'left turn', 32: 'shortcut', 33: 'Jongno', 34: 'method',
             35: 'hospital', 36: 'find', 37: 'road', 38: 'missing', 39: 'bank', 40: 'lose',
             41: 'air conditioner', 42: 'locker', 43: 'defective', 44: 'broken', 45: 'well',
             46: 'use', 47: 'impossible', 48: 'airport', 49: 'alley', 50: 'okay', 51: 'City Hall',
             52: 'turn on', 53: 'vending machine', 54: 'die'}

LENGTH = 30
current_sequence = []
predicted_sentence = ["Start"]
last_process_time = 0
PROCESS_INTERVAL = 1.0 / 30  # 30 FPS

# Store holistic instance per session
holistic_instances = {}

def extract_keypoints(results):
    face = [[res.x, res.y, res.z] for res in results.face_landmarks.landmark] if results.face_landmarks else [
        [None, None, None] for _ in range(468)]
    pose = [[res.x, res.y, res.z] for res in results.pose_landmarks.landmark] if results.pose_landmarks else [
        [None, None, None] for _ in range(33)]
    lh = [[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark] if results.left_hand_landmarks else [
        [None, None, None] for _ in range(21)]
    rh = [[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark] if results.right_hand_landmarks else [
        [None, None, None] for _ in range(21)]
    return face + lh + pose + rh

def get_top3(prediction):
    top3_idx = np.argsort(prediction['outputs'])[-3:][::-1]
    top3_probs = np.exp(prediction['outputs'][top3_idx]) / \
        np.sum(np.exp(prediction['outputs']))

    top3_words = []
    for idx in top3_idx:
        top3_words.append(ORD2SIGN2[idx])

    return top3_words, top3_probs


@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    holistic_instances[session_id] = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True
    )
    print(f"New client connected. Session ID: {session_id}")


def process_frame(frame_data, session_id):
    global current_sequence, predicted_sentence, last_process_time

    current_time = time.time()
    if current_time - last_process_time < PROCESS_INTERVAL:
        return None, None

    last_process_time = current_time

    try:
        # Get the holistic instance for this session
        holistic = holistic_instances.get(session_id)
        if not holistic:
            print(f"No holistic instance for session {session_id}")
            return None, None

        # Convert base64 to image
        _, encoded = frame_data.split(",", 1)
        frame = np.array(Image.open(io.BytesIO(base64.b64decode(encoded))))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Process with MediaPipe
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)
        frame.flags.writeable = True

        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_keypoints(results)
            current_sequence.append(keypoints)
            current_sequence = current_sequence[-LENGTH:]
            return None, None
        else:
            if len(current_sequence) == LENGTH:
                sequence_np = np.array(current_sequence, dtype=np.float32)
                prediction = prediction_fn(inputs=sequence_np)

                top3_words, top3_probs = get_top3(prediction)

                predictions = []
                for word, prob in zip(top3_words, top3_probs):
                    predictions.append({
                        'word': word,
                        'probability': float(prob)
                    })

                if prediction['outputs'].max() > 1:  # Confidence threshold
                    most_likely_word = top3_words[0]
                    if most_likely_word != predicted_sentence[-1]:
                        predicted_sentence.append(most_likely_word)
                    return predictions, most_likely_word

                current_sequence.clear()
                return predictions, None

        return None, None
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('frame')
def handle_frame(frame_data):
    session_id = request.sid
    predictions, final_word = process_frame(frame_data, session_id)
    if predictions:
        emit('predictions', {
            'predictions': predictions,
            'final_word': final_word,
            'sentence': ' '.join(predicted_sentence)
        })

@socketio.on('delete_word')
def handle_delete():
    global current_sequence, predicted_sentence
    if len(predicted_sentence) > 1:  # Keep "Start" in the list
        predicted_sentence.pop()
    current_sequence = []

@socketio.on('disconnect')
def handle_disconnect():
    global current_sequence, predicted_sentence
    session_id = request.sid
    if session_id in holistic_instances:
        holistic_instances[session_id].close()
        del holistic_instances[session_id]
    current_sequence = []
    predicted_sentence = ["Start"]
    print(f"Client disconnected. Session ID: {session_id}")


@socketio.on('start_session')
def handle_start_session():
    session_id = request.sid
    if session_id not in holistic_instances:
        holistic_instances[session_id] = mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            smooth_landmarks=True
        )
    emit('session_started', {'status': 'success'})

if __name__ == '__main__':
    socketio.run(app, debug=True)
