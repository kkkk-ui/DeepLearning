from bing_image_downloader import downloader

signs = ["止まれ 標識", "徐行 標識", "駐車禁止 標識", "一方通行 標識", "進入禁止 標識"]

for sign in signs:
    downloader.download(
        sign,
        limit=50,             # 取得する枚数
        output_dir="dataset/train",  # 保存先
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

print("✅ ダウンロード完了")
