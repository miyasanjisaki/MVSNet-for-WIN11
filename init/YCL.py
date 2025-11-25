import os
import cv2
from tqdm import tqdm


input_dir = r"C:\Users\SA_LINK\Desktop\111"      # 原始图片目录
output_dir = r"C:\Users\SA_LINK\Desktop\640512"  # 预处理后输出目录
target_w = 640
target_h = 512

os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.png')])

for img_file in tqdm(image_files, desc="Resizing images"):
    input_path = os.path.join(input_dir, img_file)
    output_path = os.path.join(output_dir, img_file)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Warning: cannot read {input_path}")
        continue

    # 缩放到目标分辨率
    resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    cv2.imwrite(output_path, resized_img)

print("All images resized to {}x{}.".format(target_w, target_h))
