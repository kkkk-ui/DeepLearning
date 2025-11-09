# ==============================
# STEP 1. 依存ライブラリ
# ==============================
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# STEP 2. データ準備
# ==============================
train_dir = "DeepLearning/dataset/augmented"
test_dir = "DeepLearning/dataset/test"

class_names  = [
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



datagen = ImageDataGenerator(
    rescale=1./255,        # 正規化
    validation_split=0.2,  # train:val = 8:2 に分割
    horizontal_flip=False
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    classes=class_names  # ← ここで順番を固定
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    classes=class_names  # ← ここで順番を固定
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # 評価時は順番を保持
)

num_classes = len(train_gen.class_indices)
print("クラス一覧:", train_gen.class_indices)

# ==============================
# STEP 3. モデル構築
# ==============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ==============================
# STEP 4. 学習
# ==============================
early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=[early]
)

loss, acc = model.evaluate(test_gen)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

class_labels = {v: k for k, v in train_gen.class_indices.items()}

img_path = input("画像ファイルのパスを入力してください: ").strip()
frame = cv2.imread(img_path)
frame_resized = cv2.resize(frame, (64, 64)) / 255.0
preds = model.predict(np.expand_dims(frame_resized, axis=0), verbose=1)
top_idx = np.argmax(preds)
print(class_labels[top_idx], preds[0][top_idx])

# ==============================
# STEP 5. モデル保存
# ==============================
model.save("traffic_sign_model.h5")
print("✅ モデルを保存しました")
