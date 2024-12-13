from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import numpy as np
import tensorflow as tf

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_fold0_55.tflite")
prediction_fn = interpreter.get_signature_runner("serving_default")

MIN_LENGTH = 20
sequence = []

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

current_sequences = {}
predicted_sentences = {}


def format_keypoints(keypoint_data):
    # Convert the received keypoint data into the format expected by the model
    face = [[point.get('x'), point.get('y'), point.get('z')] for point in keypoint_data['face']
            ] if keypoint_data['face'] else [[None, None, None] for _ in range(468)]
    pose = [[point.get('x'), point.get('y'), point.get('z')] for point in keypoint_data['pose']
            ] if keypoint_data['pose'] else [[None, None, None] for _ in range(33)]
    lh = [[point.get('x'), point.get('y'), point.get('z')] for point in keypoint_data['leftHand']
          ] if keypoint_data['leftHand'] else [[None, None, None] for _ in range(21)]
    rh = [[point.get('x'), point.get('y'), point.get('z')] for point in keypoint_data['rightHand']
          ] if keypoint_data['rightHand'] else [[None, None, None] for _ in range(21)]
    return face + lh + pose + rh


def get_top3(prediction):
    top3_idx = np.argsort(prediction['outputs'])[-3:][::-1]
    top3_probs = np.exp(prediction['outputs'][top3_idx]) / \
        np.sum(np.exp(prediction['outputs']))

    top3_words = []
    for idx in top3_idx:
        top3_words.append(ORD2SIGN2[idx])

    return top3_words, top3_probs


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    current_sequences[session_id] = []
    predicted_sentences[session_id] = []
    print(f"New client connected. Session ID: {session_id}")


@socketio.on('keypoints')
def handle_keypoints(keypoint_data):
    global sequence
    session_id = request.sid

    # Format keypoints
    keypoints = format_keypoints(keypoint_data)

    # Process if hands are present in the frame
    if any(keypoint_data['leftHand']) or any(keypoint_data['rightHand']):

        sequence.append(keypoints)
        sequence_cut = sequence[-MIN_LENGTH:]

        # 이하 코드는 실시간으로 보이는 확률을 화면에 보여주기 위해 쓴 부분
        sequence_np = np.array(sequence_cut, dtype=np.float32)
        prediction = prediction_fn(inputs=sequence_np)

        top3_words, top3_probs = get_top3(prediction)

        predictions = []
        for word, prob in zip(top3_words, top3_probs):
            predictions.append({
                'word': word,
                'probability': float(prob)
            })

        emit('predictions', {
            'predictions': predictions,
            'hands_present': True
        })

    # Predict the word if hand is not in the frame
    else:

        # 최소 프레임 길이를 넘으면 전체 sequence를 prediction function에 넣어서 예측 후 문장에 추가
        if len(sequence) >= MIN_LENGTH:
            sequence_np = np.array(sequence, dtype=np.float32)
            prediction = prediction_fn(inputs=sequence_np)
            most_likely_word = ORD2SIGN2[prediction['outputs'].argmax()]

            # Thresholding
            if prediction['outputs'].max() > 1.5:

                if not predicted_sentences[session_id] or most_likely_word != predicted_sentences[session_id][-1]:

                    predicted_sentences[session_id].append(most_likely_word)

                    emit('predictions', {
                        'final_word': most_likely_word,
                        'sentence': ' '.join(predicted_sentences[session_id])
                    })

        sequence.clear()  # sequence 제거

        # Emit event indicating no hands present
        emit('predictions', {
            'hands_present': False
        })


@socketio.on('delete_word')
def handle_delete():
    session_id = request.sid
    if session_id in predicted_sentences and len(predicted_sentences[session_id]) > 1:
        predicted_sentences[session_id].pop()

        # After popping element from the list, join them into string
        emit('predictions', {
            'sentence': ' '.join(predicted_sentences[session_id])
        })

    if session_id in current_sequences:
        current_sequences[session_id] = []


@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in current_sequences:
        del current_sequences[session_id]
    if session_id in predicted_sentences:
        del predicted_sentences[session_id]
    print(f"Client disconnected. Session ID: {session_id}")


if __name__ == '__main__':
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True,
                 host="0.0.0.0", port=5000)
