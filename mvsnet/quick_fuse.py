# quick_fuse.py  —— 纯 Python 将 MVSNet 深度融合成点云（ASCII PLY）
# 用法示例：
#   python quick_fuse.py --dense_folder E:\COL2 --depth_folder depth_est --prob_threshold 0.7 --step 2 --voxel 0.0 --out E:\COL2\mvsnet_naive.ply

import os, sys, argparse, math
import numpy as np
from PIL import Image

def load_pfm(path):
    # 兼容 text/binary PFM（灰度/彩色）
    with open(path, 'rb') as f:
        header = f.readline().decode('ascii').rstrip()
        if header not in ('PF', 'Pf'):
            raise ValueError('Not a PFM file.')
        color = header == 'PF'
        dims = f.readline().decode('ascii').strip()
        while dims.startswith('#'):  # 跳过注释行
            dims = f.readline().decode('ascii').strip()
        w, h = map(int, dims.split())
        scale = float(f.readline().decode('ascii').strip())
        data = np.fromfile(f, '<f' if scale < 0 else '>f')
        if color:
            data = np.reshape(data, (h, w, 3))
        else:
            data = np.reshape(data, (h, w))
        data = np.flipud(data)  # PFM 从底行开始
        return data

def parse_cam_txt(cam_path):
    # MVSNet cams/00000000_cam.txt: extrinsic 4x4 + intrinsic 3x3 + depth params
    with open(cam_path, 'r') as f:
        lines = [l.strip() for l in f.readlines()]
    # extrinsic 在第2~5行
    extr = np.array([[float(x) for x in lines[i].split()] for i in range(1, 5)], dtype=np.float32)
    # intrinsic 在第8~10行
    intr = np.array([[float(x) for x in lines[i].split()] for i in range(7, 10)], dtype=np.float32)
    # 深度范围（可选）
    try:
        dmin, dinterval, ndepth = [float(x) for x in lines[11].split()]
    except:
        dmin = dinterval = ndepth = None
    return intr, extr, (dmin, dinterval, ndepth)

def backproject(depth, K, extr):
    """ 将像素网格 + 深度反投影到世界坐标
        输入:
          depth: (H,W) 深度（与相机坐标系前向一致，单位米等）
          K: (3,3) 内参
          extr: (4,4) 从世界到相机的矩阵 [R|t; 0 0 0 1]
        返回:
          XYZ_world: (N,3)
    """
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    ones = np.ones_like(xs)
    pix = np.stack([xs, ys, ones], axis=-1).reshape(-1, 3).T  # (3,N)

    Kinv = np.linalg.inv(K)
    rays = Kinv @ pix  # 相机归一化射线 (3,N)
    z = depth.reshape(-1)  # (N,)
    valid = z > 0
    rays = rays[:, valid] * z[valid]  # 相机坐标 (3,Nv)

    # 相机到世界： extr 是 [R|t] 将世界->相机
    # 所以相机->世界 = extr^{-1}
    extr_inv = np.linalg.inv(extr)
    xyz_cam = np.vstack([rays, np.ones((1, rays.shape[1]))])  # (4,Nv)
    xyz_world = extr_inv @ xyz_cam  # (4,Nv)
    xyz_world = (xyz_world[:3, :] / xyz_world[3, :]).T  # (Nv,3)
    mask = valid.reshape(-1)
    return xyz_world, mask

def write_ply_ascii(path, xyz, rgb=None):
    n = xyz.shape[0]
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        if rgb is not None:
            f.write("element vertex %d\n" % n)
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(xyz, rgb):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            f.write("element vertex %d\n" % n)
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for (x, y, z) in xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

def voxel_downsample(xyz, voxel):
    if voxel <= 0:
        return xyz, None
    # 简单体素采样：取体素网格的第一个点
    q = np.floor(xyz / voxel)
    # 用字典去重
    uniq = {}
    keep_idx = []
    for i, key in enumerate(map(tuple, q.astype(np.int64))):
        if key not in uniq:
            uniq[key] = i
            keep_idx.append(i)
    keep_idx = np.array(keep_idx, dtype=np.int64)
    return xyz[keep_idx], keep_idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dense_folder', required=True, help='比如 E:\\COL2')
    ap.add_argument('--depth_folder', default='depth_est', help='深度/概率所在的子目录名（里面有 *_init.pfm, *_prob.pfm）')
    ap.add_argument('--prob_threshold', type=float, default=0.7)
    ap.add_argument('--step', type=int, default=2, help='像素步长（>1 可加速/降采样）')
    ap.add_argument('--voxel', type=float, default=0.0, help='体素大小（单位同坐标系），0 表示不下采样')
    ap.add_argument('--out', required=True, help='输出 ply 文件路径')
    args = ap.parse_args()

    root = os.path.normpath(args.dense_folder)
    img_dir = os.path.join(root, 'images')
    cam_dir = os.path.join(root, 'cams')
    dep_dir = os.path.join(root, args.depth_folder)

    # 收集影像 id（按你之前的 8 位命名，如 00000401）
    # 以深度文件为准
    ids = []
    for fn in os.listdir(dep_dir):
        if fn.endswith('_init.pfm') and len(fn) >= 12:
            stem = fn[:-9]   # 去掉 '_init.pfm'
            ids.append(stem)
    ids = sorted(ids)

    all_xyz = []
    all_rgb = []

    for sid in ids:
        img_path = os.path.join(img_dir, f'{sid}.jpg')
        cam_path = os.path.join(cam_dir, f'{sid}_cam.txt')
        dep_path = os.path.join(dep_dir, f'{sid}_init.pfm')
        prob_path = os.path.join(dep_dir, f'{sid}_prob.pfm')

        if not (os.path.exists(img_path) and os.path.exists(cam_path) and os.path.exists(dep_path) and os.path.exists(prob_path)):
            continue

        # 读取
        depth = load_pfm(dep_path).astype(np.float32)
        prob  = load_pfm(prob_path).astype(np.float32)
        K, extr, _ = parse_cam_txt(cam_path)
        img = np.array(Image.open(img_path).convert('RGB'))

        H, W = depth.shape
        # 概率筛掉低置信
        depth = np.where(prob >= args.prob_threshold, depth, 0)

        # 采样步长
        step = max(1, int(args.step))
        depth_s = depth[0:H:step, 0:W:step]
        prob_s  = prob[0:H:step, 0:W:step]
        img_s   = img[0:H:step, 0:W:step, :]

        # 反投影
        xyz, mask = backproject(depth_s, K, extr)

        # 同步颜色
        rgb = img_s.reshape(-1, 3)[mask]

        # 去除无效/NaN/inf
        ok = np.isfinite(xyz).all(axis=1)
        xyz = xyz[ok]
        rgb = rgb[ok]

        all_xyz.append(xyz)
        all_rgb.append(rgb)

        print(f"[{sid}] +{len(xyz)} pts (step={step}, prob>={args.prob_threshold})")

    if not all_xyz:
        print("没有生成任何点，请检查路径/文件命名/阈值。")
        sys.exit(1)

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    # 体素下采样（可选）
    if args.voxel > 0:
        xyz_ds, keep = voxel_downsample(xyz, args.voxel)
        rgb_ds = rgb[keep]
        print(f"voxel={args.voxel}: {len(xyz)} → {len(xyz_ds)}")
        xyz, rgb = xyz_ds, rgb_ds

    # 写出 PLY
    out_path = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_ply_ascii(out_path, xyz, rgb)
    print(f"Saved: {out_path}  ({len(xyz)} pts)")

if __name__ == '__main__':
    main()
