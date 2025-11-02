import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# モデル読み込み
# ==============================
model = load_model("traffic_sign_model.h5")

# クラス名をモデル学習時の出力順に合わせる
classes = ["STOP", "SLOW DOWN", "NO PARKING", "ONE WAY", "NO ENTRY"]

# ==============================
# Webカメラ起動
# ==============================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 標識部分を切り出して使う（最初はフレーム全体でもOK）
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)
    conf = np.max(pred)

    label = classes[class_id]
    text = f"{label} ({conf*100:.1f}%)"

    # 結果を画面に表示
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("Traffic Sign Recognition", frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

