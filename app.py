# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model

# # ==============================
# # モデル読み込み
# # ==============================
# model = load_model("traffic_sign_model.h5")

# # クラス名をモデル学習時の出力順に合わせる
# classes = ["STOP", "SLOW DOWN", "NO PARKING", "ONE WAY", "NO ENTRY"]

# # ==============================
# # Webカメラ起動
# # ==============================
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # 標識部分を切り出して使う（最初はフレーム全体でもOK）
#     img = cv2.resize(frame, (64, 64))
#     img = img / 255.0
#     img = np.expand_dims(img, axis=0)

#     pred = model.predict(img)
#     class_id = np.argmax(pred)
#     conf = np.max(pred)

#     label = classes[class_id]
#     text = f"{label} ({conf*100:.1f}%)"

#     # 結果を画面に表示
#     cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
#                 1, (0,255,0), 2, cv2.LINE_AA)

#     cv2.imshow("Traffic Sign Recognition", frame)

#     # qキーで終了
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# モデルとクラス
# ==============================
model = load_model("traffic_sign_model.h5")
classes = ["STOP", "SLOW DOWN", "NO PARKING", "ONE WAY", "NO ENTRY"]

# ==============================
# 検出関数
# ==============================
def detect_signs(frame, model, classes, window_size=64, stride=32, threshold=0.8):
    h, w, _ = frame.shape
    detections = []

    # スライドウィンドウ走査
    for y in range(0, h - window_size, stride):
        for x in range(0, w - window_size, stride):
            patch = frame[y:y+window_size, x:x+window_size]
            patch_resized = cv2.resize(patch, (64, 64)) / 255.0
            pred = model.predict(np.expand_dims(patch_resized, axis=0), verbose=0)
            class_id = np.argmax(pred)
            conf = np.max(pred)

            if conf > threshold:  # 信頼度が閾値以上なら検出
                detections.append((x, y, x+window_size, y+window_size, class_id, conf))

    return detections


# ==============================
# Webカメラ検出ループ
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_signs(frame, model, classes, window_size=64, stride=32, threshold=0.8)

    for (x1, y1, x2, y2, cls_id, conf) in detections:
        label = f"{classes[cls_id]} {conf*100:.1f}%"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imshow("Traffic Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
