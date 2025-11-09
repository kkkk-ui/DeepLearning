from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os
import numpy as np
import math

# 元画像フォルダと出力フォルダ
input_dir = "DeepLearning/dataset/train"
output_dir = "DeepLearning/dataset/augmented"

os.makedirs(output_dir, exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=10,        # 回転 ±20°
    width_shift_range=0.1,    # 横方向シフト
    height_shift_range=0.1,   # 縦方向シフト
    shear_range=0.1,         # せん断
    zoom_range=0.2,           # 拡大・縮小
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,    # 標識は左右反転しない方がよい
    fill_mode='nearest'       # 余白の補完
)

# 日本語ラベル → 英語ラベルの対応辞書
label_map = {
    "一方通行 標識 日本": "One-way_Sign",
    "動物注意 標識 日本": "Animal_Crossing_Sign",
    "右折専用 標識 日本": "Right_Turn_Only_Sign",
    "学校あり 標識 日本": "School_Zone_Sign",
    "左折専用 標識 日本": "Left_Turn_Only_Sign",
    "徐行 標識 日本": "Slow_Sign",
    "横断歩道 標識 日本": "Pedestrian_Crossing_Sign",
    "止まれ 標識 日本": "Stop_Sign",
    "直進専用 標識 日本": "Straight_Only_Sign",
    "速度制限 30 標識 日本": "Speed_Limit_30_Sign",
    "速度制限 50 標識 日本": "Speed_Limit_50_Sign",
    "速度制限 60 標識 日本": "Speed_Limit_60_Sign",
    "進入禁止 標識 日本": "No_Entry_Sign",
    "駐停車禁止 標識 日本": "No_Parking_or_Stopping_Sign",
    "駐車禁止 標識 日本": "No_Parking_Sign"
}

# 各クラスごとに処理
for label in os.listdir(input_dir):
    class_dir = os.path.join(input_dir, label)
    save_dir = os.path.join(output_dir, label_map[label])  # 英語ラベルに変換
    os.makedirs(save_dir, exist_ok=True)

    images = os.listdir(class_dir)
    num_images = len(images)
    
    # 元画像1枚あたり何枚増やすか計算（端数は切り上げ）
    aug_per_image = math.ceil(300 / num_images)

    for file in images:
        img_path = os.path.join(class_dir, file)
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, 0)

        # 画像を aug_per_image 枚生成
        i = 0
        for batch in datagen.flow(
                x, batch_size=1, save_to_dir=save_dir,
                save_prefix="aug", save_format="jpg"):
            i += 1
            if i >= aug_per_image:
                break


print("✅ 画像の拡張が完了しました！")
