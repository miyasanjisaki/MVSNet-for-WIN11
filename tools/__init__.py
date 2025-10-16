import os
import numpy as np


COLMAP_DIR = r"E:\COL"
COLMAP_CAMERAS_TXT = os.path.join(COLMAP_DIR, "cameras.txt")
COLMAP_IMAGES_TXT = os.path.join(COLMAP_DIR, "images.txt")
FRAMES_DIR = r"C:\Users\SA_LINK\Desktop\c60536c727ccdb94658e9a0c515a69f2_frames"
OUTPUT_DIR = os.path.join(COLMAP_DIR, "MVSNet")
OUTPUT_CAM_DIR = os.path.join(OUTPUT_DIR, "cams")
OUTPUT_PAIR_FILE = os.path.join(OUTPUT_DIR, "pair.txt")
SOURCE_VIEW_NUM = 5  # 每帧使用前后多少张作为 source view

os.makedirs(OUTPUT_CAM_DIR, exist_ok=True)


def parse_cameras(file_path):
    cameras = {}
    with open(file_path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))
            cameras[cam_id] = {
                'model': model,
                'width': width,
                'height': height,
                'params': params
            }
    return cameras


def parse_images(file_path):
    images = {}
    with open(file_path) as f:
        lines = f.readlines()
    for i in range(0, len(lines), 2):
        line = lines[i].strip()
        if line.startswith('#') or line == '':
            continue
        parts = line.split()
        img_id = int(parts[0])
        qx, qy, qz, qw = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        img_name = parts[9]
        images[img_name] = {
            'cam_id': cam_id,
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            't': np.array([tx, ty, tz])
        }
    return images


def quat2mat(qw, qx, qy, qz):
    R = np.zeros((3, 3))
    R[0,0] = 1 - 2*qy**2 - 2*qz**2
    R[0,1] = 2*qx*qy - 2*qz*qw
    R[0,2] = 2*qx*qz + 2*qy*qw
    R[1,0] = 2*qx*qy + 2*qz*qw
    R[1,1] = 1 - 2*qx**2 - 2*qz**2
    R[1,2] = 2*qy*qz - 2*qx*qw
    R[2,0] = 2*qx*qz - 2*qy*qw
    R[2,1] = 2*qy*qz + 2*qx*qw
    R[2,2] = 1 - 2*qx**2 - 2*qy**2
    return R


cameras = parse_cameras(COLMAP_CAMERAS_TXT)
images = parse_images(COLMAP_IMAGES_TXT)

frame_names = sorted([f for f in os.listdir(FRAMES_DIR) if f.lower().endswith(('.png','.jpg','.jpeg'))])

with open(os.path.join(OUTPUT_CAM_DIR, "0000_cam.txt"), 'w') as f:
    for fname in frame_names:
        img_info = images[fname]
        cam_info = cameras[img_info['cam_id']]
        fx, fy, cx, cy = cam_info['params'][0], cam_info['params'][1], cam_info['params'][2], cam_info['params'][3]
        R = quat2mat(img_info['qw'], img_info['qx'], img_info['qy'], img_info['qz'])
        t = img_info['t']
        # 按 MVSNet 格式写入
        f.write(f"{fname} {fx} 0 {cx} 0 {fy} {cy} 0 0 1 " +
                " ".join([str(R[i,j]) for i in range(3) for j in range(3)]) + " " +
                " ".join([str(t[i]) for i in range(3)]) + "\n")

print("cams/0000_cam.txt 已生成")

with open(OUTPUT_PAIR_FILE, 'w') as f:
    f.write(f"{len(frame_names)}\n")  # 总帧数
    for idx, fname in enumerate(frame_names):
        f.write(f"{idx}\n")  # 参考帧编号
        # 选择前后 SOURCE_VIEW_NUM 张作为 source views
        src_idxs = []
        for i in range(1, SOURCE_VIEW_NUM+1):
            if idx-i >=0:
                src_idxs.append(idx-i)
            if idx+i < len(frame_names):
                src_idxs.append(idx+i)
        f.write(f"{len(src_idxs)} {' '.join(map(str, src_idxs))}\n")

print("pair.txt 已生成")
print(f"输出目录: {OUTPUT_DIR}")
