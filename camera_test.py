import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# ==========================================================
# 【変更点】
# mediapipe 0.10.31以降、Windows/Python3.13環境などで
# `mp.solutions` がロードに失敗する不具合が報告されています
# (Google公式でも旧solutions APIは非推奨・開発終了扱い)。
# そのため mp.solutions.hands の代わりに、
# 新しい Tasks API (HandLandmarker) を使用します。
#
#   curl -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
#
# （curlが無ければブラウザで上記URLを開いて保存してもOKです）
# ==========================================================

MODEL_PATH = "./models/hand_landmarker.task"

# MediaPipe公式の21点手指ランドマークの接続定義（solutions.HAND_CONNECTIONS相当）
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 親指
    (0, 5), (5, 6), (6, 7), (7, 8),          # 人差し指
    (5, 9), (9, 10), (10, 11), (11, 12),     # 中指
    (9, 13), (13, 14), (14, 15), (15, 16),   # 薬指
    (13, 17), (17, 18), (18, 19), (19, 20),  # 小指
    (0, 17),                                 # 手のひら
]


def draw_landmarks(image, hand_landmarks_list):
    """検出結果を画像に手描きで描画する（drawing_utils非依存）"""
    h, w, _ = image.shape
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(image, points[start_idx], points[end_idx], (255, 255, 255), 2)

        for x, y in points:
            cv2.circle(image, (x, y), 4, (0, 200, 0), -1)


def main():
    base_options = BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        print("カメラを起動しました。終了するには 'q' を押してください。")

        frame_timestamp_ms = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 左右反転（鏡のように表示するため）
            image = cv2.flip(image, 1)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # 動画モードではフレームごとにタイムスタンプが必要（単調増加であればOK）
            frame_timestamp_ms += 33
            result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            if result.hand_landmarks:
                draw_landmarks(image, result.hand_landmarks)

            cv2.imshow("MediaPipe Hands", image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
