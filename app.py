from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import mediapipe as mp

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions and model details
DATA_PATH = os.path.join('MP_Data')  # Path to your dataset
actions = np.array([
   "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "Enye", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", 
    "Magandang Hapon", "Kumusta ka na", "Ako ay mabuti", "Anong ginagawa mo", "buntis", "aral", "gusto", "magluto", "maganda", "kahapon", "Lunes", "Enero",
    "Pula", "Restawran", "Kondominyum", "Gutom", "Kumain", "Saging", "Pancit", "Inihaw na bangus", "Motorsiklo", "Paniki", "Sakit ng ulo", "Pagkahilo", "Takdang-Aralin", 
    "Ah ganun", "Ano pangalan mo", "Hanggang sa muli", "Mahal kita", "Salamat", "Ano", "Saan", "Bakit", "Magkano", "Manila", "Ako", "At", "Isang daan"
])
sequence_length = 40
model = None  # Placeholder for the trained model
predicted_text = ""  # Store the current prediction


# Load dataset from MP_Data
def load_data():
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        action_path = os.path.join(DATA_PATH, action)
        for sequence in os.listdir(action_path):
            window = []
            for frame_num in range(sequence_length):
                frame_path = os.path.join(action_path, sequence, f"{frame_num}.npy")
                res = np.load(frame_path)
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])
    return np.array(sequences), to_categorical(labels).astype(int)


# Build LSTM model
def build_model():
    global model
    model = Sequential()
    model.add(Input(shape=(40, 126)))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# Train the model
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
    build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)
    model.save_weights('action.h5')  # Save trained weights


# Load the trained model
def load_trained_model():
    global model
    build_model()
    model.load_weights('action.h5')


# Extract keypoints from Mediapipe results
def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([lh, rh])


# Video feed with predictions
def generate_frames():
    global predicted_text
    if model is None:
        print("Model is not initialized. Building and loading weights...")
        load_trained_model()

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        sequence = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            image, results = mediapipe_detection(frame, holistic)
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Extract keypoints
            keypoints = extract_keypoints(results)
            print(f"Extracted keypoints shape: {keypoints.shape}")  # Debug: Keypoint shape
            sequence.append(keypoints)
            sequence = sequence[-40:]  # Keep only the last 30 frames

            if len(sequence) == 40:
                try:
                    print(f"Sequence shape: {np.array(sequence).shape}")  # Debug: Sequence shape
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(f"Prediction result: {res}")  # Debug: Prediction output
                    if res[np.argmax(res)] > 0.9:
                        predicted_text = actions[np.argmax(res)]
                        print(f"Predicted text: {predicted_text}")  # Debug: Final prediction
                except Exception as e:
                    print(f"Error during prediction: {e}")

            # Encode frame for video streaming
            _, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Mediapipe detection helper
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


# Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    train_model()
    return "Model training completed!"


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/translation')
def translation():
    return jsonify({"text": predicted_text})


if __name__ == '__main__':
    if not os.path.exists('action.h5'):
        train_model()  # Train the model if no pre-trained weights exist
    else:
        load_trained_model()  # Load pre-trained weights
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 10000)), debug=True)


