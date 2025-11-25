import cv2
import os


video_path = r"E:\2025-11-23_202910_671.mp4"
output_dir = r"C:\Users\SA_LINK\Desktop\111"
frames_per_second = 5


os.makedirs(output_dir, exist_ok=True)


cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception(f"无法打开视频: {video_path}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"视频帧率: {video_fps} FPS, 总帧数: {frame_count}")


save_interval = int(round(video_fps / frames_per_second))
print(f"每隔 {save_interval} 帧保存一张图片")

frame_idx = 0
saved_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % save_interval == 0:
        filename = os.path.join(output_dir, f"{saved_idx:04d}.png")
        cv2.imwrite(filename, frame)
        saved_idx += 1

    frame_idx += 1

cap.release()
print(f"完成！总共保存 {saved_idx} 张图片，保存在 {output_dir}")
