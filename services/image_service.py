import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "finetuned_emotion_cnn.keras",
    )
)

CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

IMAGE_SIZE = (96, 96)

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"Emotion model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False

print("LOADED MODEL:", MODEL_PATH)
print("INPUT SHAPE:", model.input_shape)
print("OUTPUT SHAPE:", model.output_shape)

CASCADE_PATH = os.path.join(
    cv2.data.haarcascades,
    "haarcascade_frontalface_default.xml",
)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)


def analyze_image(image_path):
    """Detect the largest face in an image and predict its emotion.

    Falls back to the whole image if no face is detected.

    Returns a dict with the predicted label and per-class
    probabilities (in percent).
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        rgb_image = np.asarray(image)

    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    faces = face_detector.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    if len(faces) > 0:
        # Pick the largest detected face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Add a 10% margin around the face, clipped to image bounds
        margin_x = int(w * 0.10)
        margin_y = int(h * 0.10)

        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(rgb_image.shape[1], x + w + margin_x)
        y2 = min(rgb_image.shape[0], y + h + margin_y)

        face = rgb_image[y1:y2, x1:x2]
        face_gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
    else:
        face_gray = gray_image

    face_gray = cv2.resize(
        face_gray,
        IMAGE_SIZE,
        interpolation=cv2.INTER_LINEAR,
    )

    # Replicate grayscale into 3 channels and normalize to [0, 1]
    image_array = np.stack([face_gray] * 3, axis=-1).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(image_array, verbose=0)[0]
    prediction_id = int(np.argmax(probabilities))

    return {
        "prediction": CLASS_NAMES[prediction_id],
        "probabilities": {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
    }