from bing_image_downloader import downloader

signs = [
    "止まれ 標識", "徐行 標識", "駐車禁止 標識", "一方通行 標識", "進入禁止 標識",
    "右折専用 標識", "左折専用 標識", "直進専用 標識", "駐停車禁止 標識",
    "横断歩道 標識", "学校あり 標識", "動物注意 標識",
    "速度制限 30 標識", "速度制限 50 標識", "速度制限 60 標識"
]

for sign in signs:
    downloader.download(
        sign + " 日本",
        limit=30,             # 取得する枚数
        output_dir="DeepLearning/dataset/train",  # 保存先
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("✅ ダウンロード完了")
