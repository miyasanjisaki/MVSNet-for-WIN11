"""
export_mvsnet_images.py
—— 从 COLMAP 的 images.txt 导出连续编号图片（适配 MVSNet）
"""

import os
import cv2

IMAGE_DIR = r"E:\COL\images"                # COLMAP 原始图片路径
IMAGES_TXT = r"E:\COL\images.txt"  # COLMAP 导出的 images.txt
OUTPUT_DIR = r"E:\COL\images2"        # 导出目录


def read_colmap_image_names(txt_path):
    """只读取 images.txt 第一行的图片名（跳过2D点行）"""
    names = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.strip().split()
            # 只有第一行包含 >=10 个字段，第二行是坐标行
            if len(parts) >= 10:
                name = parts[-1]
                names.append(name)
    print(f"读取到 {len(names)} 张注册图片。")
    return names

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    names = read_colmap_image_names(IMAGES_TXT)

    print(f"开始导出图片到 {OUTPUT_DIR} ...")
    for idx, name in enumerate(names):
        src_path = os.path.join(IMAGE_DIR, name)
        dst_path = os.path.join(OUTPUT_DIR, f"{idx:08d}.jpg")

        if not os.path.exists(src_path):
            print(f" 找不到原图: {src_path}")
            continue

        img = cv2.imread(src_path)
        if img is None:
            print(f" 无法读取图像: {src_path}")
            continue

        cv2.imwrite(dst_path, img)
        if idx % 50 == 0:
            print(f"进度: {idx}/{len(names)}")

    print(f"导出完成，共 {len(names)} 张图片。")

if __name__ == "__main__":
    main()
