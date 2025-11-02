# ==============================
# STEP 1. 依存ライブラリ
# ==============================
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ==============================
# STEP 2. データ準備
# ==============================
train_dir = "dataset/train"

datagen = ImageDataGenerator(
    rescale=1./255,        # 正規化
    validation_split=0.2,  # train:val = 8:2 に分割
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2]
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_gen.class_indices)
print("クラス一覧:", train_gen.class_indices)

# ==============================
# STEP 3. モデル構築
# ==============================
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
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
early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[early]
)

# ==============================
# STEP 5. モデル保存
# ==============================
model.save("traffic_sign_model.h5")
print("✅ モデルを保存しました")
