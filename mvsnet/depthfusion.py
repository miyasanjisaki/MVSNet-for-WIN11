#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Convert MVSNet output to Gipuma format for post-processing.
"""

from __future__ import print_function

import argparse
import os
import time
import glob
import random
import math
import re
import sys
import shutil
from struct import *

import cv2
import numpy as np

def load_pfm_path(path):
    with open(path, "rb") as f:
        return load_pfm(f)

import pylab as plt
from preprocess import * 

def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]
        
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''
    
    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return 

def mvsnet_to_gipuma_dmb(in_path, out_path):
    '''convert mvsnet .pfm output to Gipuma .dmb format'''
    
    image = load_pfm_path(in_path)
    write_gipuma_dmb(out_path, image)

    return 

def mvsnet_to_gipuma_cam(in_path, out_path):
    '''convert mvsnet camera to gipuma camera format'''

    cam = load_cam(open(in_path))

    extrinsic = cam[0:4][0:4][0]
    intrinsic = cam[0:4][0:4][1]
    intrinsic[3][0] = 0
    intrinsic[3][1] = 0
    intrinsic[3][2] = 0
    intrinsic[3][3] = 0
    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:]
    
    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return

def fake_gipuma_normal(in_depth_path, out_normal_path):
    
    depth_image = read_gipuma_dmb(in_depth_path)
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)
    return 

def mvsnet_to_gipuma(dense_folder, gipuma_point_folder, depth_folder_name):
    """按深度目录里的前缀驱动（8 位/4 位都可），相机从 depth 目录取 .txt，
       颜色图优先从 images/ 找；若不存在则从 depth 目录找同名 .jpg/.png"""
    image_folder = os.path.join(dense_folder, "images")
    depth_folder = os.path.join(dense_folder, depth_folder_name)

    gipuma_cam_folder = os.path.join(gipuma_point_folder, "cams")
    gipuma_image_folder = os.path.join(gipuma_point_folder, "images")
    os.makedirs(gipuma_point_folder, exist_ok=True)
    os.makedirs(gipuma_cam_folder, exist_ok=True)
    os.makedirs(gipuma_image_folder, exist_ok=True)

    # 以 *_prob_filtered.pfm 为准；没有的话退回 *_init.pfm
    pfm_list = sorted(glob.glob(os.path.join(depth_folder, "*_prob_filtered.pfm")))
    if not pfm_list:
        pfm_list = sorted(glob.glob(os.path.join(depth_folder, "*_init.pfm")))
    if not pfm_list:
        raise FileNotFoundError(f"No *_prob_filtered.pfm or *_init.pfm under {depth_folder}")

    prefixes = [os.path.basename(p)[:-len("_prob_filtered.pfm")] if p.endswith("_prob_filtered.pfm")
                else os.path.basename(p)[:-len("_init.pfm")] for p in pfm_list]

    # 相机：depth 目录下 <prefix>.txt
    for prefix in prefixes:
        in_cam_file = os.path.join(depth_folder, prefix + ".txt")
        out_cam_file = os.path.join(gipuma_cam_folder, prefix + ".jpg.P")  # 原脚本以 image_name+'.P' ，我们统一用 <prefix>.jpg.P
        if not os.path.exists(in_cam_file):
            continue
        mvsnet_to_gipuma_cam(in_cam_file, out_cam_file)

    # 颜色图：优先 images/<prefix>.{jpg,png}; 否则 depth/<prefix>.{jpg,png}
    for prefix in prefixes:
        src_img = None
        for ext in (".jpg", ".png", ".jpeg"):
            cand = os.path.join(image_folder, prefix + ext)
            if os.path.exists(cand):
                src_img = cand; break
        if src_img is None:
            for ext in (".jpg", ".png", ".jpeg"):
                cand = os.path.join(depth_folder, prefix + ext)
                if os.path.exists(cand):
                    src_img = cand; break
        if src_img is None:
            continue
        shutil.copy(src_img, os.path.join(gipuma_image_folder, os.path.basename(src_img)))

    # 深度图转换 & 生成假法线
    gipuma_prefix = "2333__"
    for prefix in prefixes:
        sub_depth_folder = os.path.join(gipuma_point_folder, gipuma_prefix + prefix)
        os.makedirs(sub_depth_folder, exist_ok=True)
        in_depth_pfm = os.path.join(depth_folder, prefix + "_prob_filtered.pfm")
        if not os.path.exists(in_depth_pfm):
            in_depth_pfm = os.path.join(depth_folder, prefix + "_init.pfm")
            if not os.path.exists(in_depth_pfm):
                continue
        out_depth_dmb = os.path.join(sub_depth_folder, "disp.dmb")
        fake_normal_dmb = os.path.join(sub_depth_folder, "normals.dmb")
        mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb)
        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)


def probability_filter(dense_folder, prob_threshold, depth_folder_name):
    """按深度目录里的 *_init.pfm 遍历，不依赖 images 命名位数"""
    depth_folder = os.path.join(dense_folder, depth_folder_name)
    init_list = sorted(glob.glob(os.path.join(depth_folder, "*_init.pfm")))
    if not init_list:
        raise FileNotFoundError(f"No *_init.pfm under {depth_folder}")

    for init_path in init_list:
        prefix = os.path.basename(init_path)[:-len("_init.pfm")]
        # 兼容多种概率图命名
        candidates = [
            os.path.join(depth_folder, prefix + "_prob.pfm"),
            os.path.join(depth_folder, prefix + "_prob_map.pfm"),
            os.path.join(depth_folder, prefix + "_confidence_map.pfm"),
        ]
        prob_path = next((p for p in candidates if os.path.exists(p)), None)
        if prob_path is None:
            # 没有概率图就跳过该视图
            continue

        out_path = os.path.join(depth_folder, prefix + "_prob_filtered.pfm")
        depth_map = load_pfm(open(init_path, "rb"))
        prob_map = load_pfm(open(prob_path, "rb"))
        depth_map[prob_map < prob_threshold] = 0
        write_pfm(out_path, depth_map)

def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):
    cam_folder = os.path.join(point_folder, "cams")
    image_folder = os.path.join(point_folder, "images")
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = (f"{fusibile_exe_path} -input_folder {point_folder}/"
            f" -p_folder {cam_folder}/ -images_folder {image_folder}/"
            f" --depth_min={depth_min} --depth_max={depth_max}"
            f" --normal_thresh={normal_thresh} --disp_thresh={disp_thresh}"
            f" --num_consistent={num_consistent}")
    print(cmd)
    os.system(cmd)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense_folder', type=str, required=True)
    parser.add_argument('--depth_folder', type=str, default='depths_mvsnet')  # 新增，可传 depth_est
    parser.add_argument('--fusibile_exe_path', type=str, default='fusibile')
    parser.add_argument('--prob_threshold', type=float, default=0.8)
    parser.add_argument('--disp_threshold', type=float, default=0.25)
    parser.add_argument('--num_consistent', type=int, default=3)
    args = parser.parse_args()

    dense_folder = args.dense_folder
    depth_folder = args.depth_folder
    fusibile_exe_path = args.fusibile_exe_path
    prob_threshold = args.prob_threshold
    disp_threshold = args.disp_threshold
    num_consistent = args.num_consistent

    point_folder = os.path.join(dense_folder, 'points_mvsnet')
    os.makedirs(point_folder, exist_ok=True)

    print('filter depth map with probability map')
    probability_filter(dense_folder, prob_threshold, depth_folder)

    print('Convert mvsnet output to gipuma input')
    mvsnet_to_gipuma(dense_folder, point_folder, depth_folder)

    print('Run depth map fusion & filter')
    depth_map_fusion(point_folder, fusibile_exe_path, disp_threshold, num_consistent)

