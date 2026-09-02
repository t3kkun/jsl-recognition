import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "models/hand_landmarker.task"
IMAGE_PATH = "testdata/hand.jpg"

# Create hand landmarker
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
)

detector = vision.HandLandmarker.create_from_options(options)

# Load image
image = mp.Image.create_from_file(IMAGE_PATH)

# Detect
result = detector.detect(image)

# Output
print(f"Hands detected: {len(result.hand_landmarks)}")

if result.hand_landmarks:
    landmarks = result.hand_landmarks[0]

    print(f"Landmarks: {len(landmarks)}")

    for i, landmark in enumerate(landmarks):
        print(
            f"{i:2d}: "
            f"x={landmark.x:.4f}, "
            f"y={landmark.y:.4f}, "
            f"z={landmark.z:.4f}"
        )