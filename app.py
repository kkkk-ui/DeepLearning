import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model

# ==============================
# モデルとクラス
# ==============================
model = load_model("traffic_sign_model.h5")
classes = [
    "One-way_Sign",                  # 一方通行 標識 日本
    "Animal_Crossing_Sign",          # 動物注意 標識 日本
    "Right_Turn_Only_Sign",          # 右折専用 標識 日本
    "School_Zone_Sign",              # 学校あり 標識 日本
    "Left_Turn_Only_Sign",           # 左折専用 標識 日本
    "Slow_Sign",                      # 徐行 標識 日本
    "Pedestrian_Crossing_Sign",      # 横断歩道 標識 日本
    "Stop_Sign",                      # 止まれ 標識 日本
    "Straight_Only_Sign",             # 直進専用 標識 日本
    "Speed_Limit_30_Sign",            # 速度制限 30 標識 日本
    "Speed_Limit_50_Sign",            # 速度制限 50 標識 日本
    "Speed_Limit_60_Sign",            # 速度制限 60 標識 日本
    "No_Entry_Sign",                  # 進入禁止 標識 日本
    "No_Parking_or_Stopping_Sign",   # 駐停車禁止 標識 日本
    "No_Parking_Sign"                 # 駐車禁止 標識 日本
]

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


import cv2

# 入力モード選択
mode = input("入力モードを選んでください（image/camera/classify）：").strip().lower()

if mode == "image":
    # ==============================
    # 画像ファイル検出
    # ==============================
    # パラメータ
    random.seed(42)
    colors = {cls_id: (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            for cls_id in range(len(classes))}
    threshold = 0.9        # CNN検出確率閾値
    min_size = 32          # 小さい領域は無視
    window_size = (80, 80) # スライディングウィンドウサイズ
    step_size = 32          # ウィンドウ移動幅

    img_path = input("画像ファイルのパスを入力してください: ").strip()
    img = cv2.imread(img_path)
    img_r = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("binary", binary)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # --- 1. 色で候補領域抽出 ---
    # 赤色
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(img_hsv, lower_red1, upper_red1),
                            cv2.inRange(img_hsv, lower_red2, upper_red2))
    
    cv2.imshow("binary", mask_red)
    cv2.imwrite("mask_red.jpg", mask_red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 青色
    lower_blue = np.array([100, 130, 60])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

    cv2.imshow("binary", mask_blue)
    cv2.imwrite("mask_blue.jpg", mask_blue)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 黄色
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    cv2.imshow("binary", mask_yellow)
    cv2.imwrite("mask_yellow.jpg", mask_yellow)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 全色マスクを統合
    mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_yellow))

    cv2.imshow("binary", mask)
    cv2.imwrite("mask.jpg", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
 
    # モルフォロジー処理
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # 収縮処理
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1) # 膨張処理

    cv2.imshow("Hybrid Detection (RGB + 3 Colors)", mask)
    cv2.imwrite("edge_mask.jpg", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 輪郭抽出
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- 2. 候補領域ごとにスライディングウィンドウ + CNN分類 ---
    print(contours)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_size or h < min_size:
            continue
        aspect_ratio = w / h
        if aspect_ratio < 0.3 or aspect_ratio > 1.6:
            continue

        roi = img[y:y+h, x:x+w]
        cv2.imshow("Hybrid Detection (RGB + 3 Colors)", roi)
        cv2.imwrite("detective.jpg", roi)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # BGR -> RGBに変換
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # CNN入力サイズへリサイズして正規化
        roi_resized = cv2.resize(roi_rgb, (64, 64))
        roi_input = roi_resized / 255.0
        roi_input = np.expand_dims(roi_input, axis=0)

        # 推論
        pred = model.predict(roi_input, verbose=0)
        conf = pred.max()
        cls_id = pred.argmax()

        # しきい値を満たす場合のみ描画
        if conf > threshold:
            color = colors[cls_id]
            label = f"{classes[cls_id]} {conf*100:.1f}%"

            cv2.rectangle(img_r, (x, y), (x+w, y+h), color, 2)
            cv2.rectangle(img_r, (x, y-20), (x+w, y), color, -1)
            cv2.putText(img_r, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # 結果表示
    cv2.imshow("Hybrid Detection (RGB + 3 Colors)", img_r)
    cv2.imwrite("img_detective.jpg", img_r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif mode == "classify":
    img_path = input("画像ファイルのパスを入力してください: ").strip()
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame, (64, 64)) / 255.0

    pred = model.predict(np.expand_dims(frame_resized, axis=0), verbose=1)
    class_id = np.argmax(pred)
    conf = np.max(pred)

    print(pred)
    print(classes[class_id])
    print(conf)

elif mode == "camera":
    # パラメータ
    random.seed(42)
    colors = {cls_id: (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            for cls_id in range(len(classes))}
    threshold = 0.9        # CNN検出確率閾値
    min_size = 32          # 小さい領域は無視
    window_size = (80, 80) # スライディングウィンドウサイズ
    step_size = 32          # ウィンドウ移動幅

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 1. 色で候補領域抽出 ---
        # 赤色
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([179, 255, 255])
        mask_red = cv2.bitwise_or(cv2.inRange(img_hsv, lower_red1, upper_red1),
                                cv2.inRange(img_hsv, lower_red2, upper_red2))

        # 青色
        lower_blue = np.array([100, 130, 60])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)

        # 黄色
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])
        mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

        # 全色マスクを統合
        mask = cv2.bitwise_or(mask_red, cv2.bitwise_or(mask_blue, mask_yellow))
    
        # モルフォロジー処理
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # 収縮処理
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1) # 膨張処理

        # 輪郭抽出
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 2. 候補領域ごとにスライディングウィンドウ + CNN分類 ---
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < min_size or h < min_size:
                continue
            aspect_ratio = w / h
            if aspect_ratio < 0.3 or aspect_ratio > 1.6:
                continue

            roi = frame[y:y+h, x:x+w]

            # BGR -> RGBに変換
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # CNN入力サイズへリサイズして正規化
            roi_resized = cv2.resize(roi_rgb, (64, 64))
            roi_input = roi_resized / 255.0
            roi_input = np.expand_dims(roi_input, axis=0)

            # 推論
            pred = model.predict(roi_input, verbose=0)
            conf = pred.max()
            cls_id = pred.argmax()

            # しきい値を満たす場合のみ描画
            if conf > threshold:
                color = colors[cls_id]
                label = f"{classes[cls_id]} {conf*100:.1f}%"

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame, (x, y-20), (x+w, y), color, -1)
                cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 結果を画面に表示  
        cv2.imshow("Traffic Sign Detection", frame)

        # qキーで終了
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()    

else:
    print("モードが正しくありません。image または camera を入力してください。")

