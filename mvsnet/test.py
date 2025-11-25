#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Test script.
"""
from __future__ import print_function

import os
# 先清掉可能已设置的值，再强制禁用
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


import os
import time
import sys
import math
import argparse
import numpy as np
from absl import flags
import multiprocessing
from tensorflow.python.lib.io import file_io


import cv2
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.append("../")
from tools.common import Notify
from preprocess import *
from model import *
from loss import *

# input path
tf.app.flags.DEFINE_string('dense_folder', None, 
                           """Root path to dense folder.""")
tf.app.flags.DEFINE_string('pretrained_model_ckpt_path', 
                           '/data/tf_model/3DCNNs/BlendedMVS/blended_augmented/model.ckpt',
                           """Path to restore the model.""")
tf.app.flags.DEFINE_integer('ckpt_step', 150000,
                            """ckpt step.""")

# input parameters
tf.app.flags.DEFINE_integer('view_num', 5,
                            """Number of images (1 ref image and view_num - 1 view images).""")
tf.app.flags.DEFINE_integer('max_d', 256, 
                            """Maximum depth step when testing.""")
tf.app.flags.DEFINE_integer('max_w', 1600, 
                            """Maximum image width when testing.""")
tf.app.flags.DEFINE_integer('max_h', 1200, 
                            """Maximum image height when 
                            
                            
                            
                            testing.""")
tf.app.flags.DEFINE_float('sample_scale', 0.25, 
                            """Downsample scale for building cost volume (W and H).""")
tf.app.flags.DEFINE_float('interval_scale', 0.8, 
                            """Downsample scale for building cost volume (D).""")
tf.app.flags.DEFINE_float('base_image_size', 8, 
                            """Base image size""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """Testing batch size.""")
tf.app.flags.DEFINE_bool('adaptive_scaling', True, 
                            """Let image size to fit the network, including 'scaling', 'cropping'""")

# network architecture
tf.app.flags.DEFINE_string('regularization', 'GRU',
                           """Regularization method, including '3DCNNs' and 'GRU'""")
tf.app.flags.DEFINE_boolean('refinement', False,
                           """Whether to apply depth map refinement for MVSNet""")
tf.app.flags.DEFINE_bool('inverse_depth', True,
                           """Whether to apply inverse depth for R-MVSNet""")

FLAGS = tf.app.flags.FLAGS



FLAGS = flags.FLAGS

# 定义你需要的命令行参数
flags.DEFINE_string('dataset', 'dtu_yao', 'Dataset name')
flags.DEFINE_string('testlist', 'lists/test.txt', 'Test list file')
flags.DEFINE_integer('numdepth', 192, 'Number of depth planes')
flags.DEFINE_string('loadckpt', 'checkpoints/mvsnet_model.ckpt', 'Checkpoint to load')


class MVSGenerator:
    """ data generator class, tf only accept generator without param """
    def __init__(self, sample_list, view_num):
        self.sample_list = sample_list
        self.view_num = view_num
        self.sample_num = len(sample_list)
        self.counter = 0

    def __iter__(self):
        while True:
            for data in self.sample_list:
                images, cams = [], []
                image_index = int(os.path.splitext(os.path.basename(data[0]))[0])
                selected_view_num = int(len(data) // 2)

                # 读取 ref + src
                for view in range(min(self.view_num, selected_view_num)):
                    # img
                    img_path = data[2 * view]
                    image = cv2.imread(img_path)  # BGR
                    if image is None:
                        raise FileNotFoundError(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 统一为 RGB
                    # cam
                    cam_file = file_io.FileIO(data[2 * view + 1], mode='r')
                    cam = load_cam(cam_file, FLAGS.interval_scale)
                    if cam[1][3][2] == 0:
                        cam[1][3][2] = FLAGS.max_d
                    images.append(image)
                    cams.append(cam)

                # 邻居不足用 ref 填充（用 data[0]/data[1]）
                if selected_view_num < self.view_num:
                    for _ in range(selected_view_num, self.view_num):
                        image = cv2.imread(data[0])
                        if image is None:
                            raise FileNotFoundError(data[0])
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        cam_file = file_io.FileIO(data[1], mode='r')
                        cam = load_cam(cam_file, FLAGS.interval_scale)
                        images.append(image)
                        cams.append(cam)

                print('range: ', cams[0][1, 3, 0], cams[0][1, 3, 1], cams[0][1, 3, 2], cams[0][1, 3, 3])

                # 自适应缩放
                resize_scale = 1.0
                if FLAGS.adaptive_scaling:
                    h_scale = max(float(FLAGS.max_h) / im.shape[0] for im in images)
                    w_scale = max(float(FLAGS.max_w) / im.shape[1] for im in images)
                    if h_scale > 1 or w_scale > 1:
                        print("max_h, max_w should < W and H!")
                        sys.exit(-1)
                    resize_scale = max(h_scale, w_scale)

                scaled_input_images, scaled_input_cams = scale_mvs_input(images, cams, scale=resize_scale)

                # 裁剪
                croped_images, croped_cams = crop_mvs_input(scaled_input_images, scaled_input_cams)

                # 中心化（注意要 stack！）
                centered_images = [center_image(croped_images[v]) for v in range(self.view_num)]
                centered_images = np.stack(centered_images, axis=0)  # (V,H,W,3)

                # 采样相机
                scaled_cams = scale_mvs_camera(croped_cams, scale=FLAGS.sample_scale)

                # 输出
                scaled_images = [scale_image(croped_images[v], scale=FLAGS.sample_scale) for v in range(self.view_num)]
                scaled_images = np.stack(scaled_images, axis=0)  # (V,h,w,3)
                croped_images = np.stack(croped_images, axis=0)  # (V,Hc,Wc,3)
                scaled_cams = np.stack(scaled_cams, axis=0)  # (V,2,4,4)
                yield (scaled_images, centered_images, scaled_cams, image_index)


# def mvsnet_pipeline(mvs_list):
#
#     """ mvsnet in altizure pipeline """
#     print ('Testing sample number: ', len(mvs_list))
#
#     # create output folder
#     output_folder = os.path.join(FLAGS.dense_folder, 'depths_mvsnet')
#     if not os.path.isdir(output_folder):
#         os.mkdir(output_folder)
#
#     # testing set
#     mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
#     generator_data_type = (tf.float32, tf.float32, tf.float32, tf.int32)
#     # 1. 生成 image_paths 和 cam_paths 的 tensor
#     image_paths = tf.constant([entry[0] for entry in mvs_list])
#     cam_paths = tf.constant([entry[1] for entry in mvs_list])
#
#     def load_sample(img_path, cam_path):
#         # img_path: string tensor
#         img = tf.io.read_file(img_path)
#         img = tf.image.decode_jpeg(img, channels=3)
#         img = tf.image.convert_image_dtype(img, tf.float32)
#
#         # cam_path 是 string tensor，需要用 tf.py_function 读取
#         def load_cam_np(path):
#             path = path.decode('utf-8')  # 从 bytes 转 str
#             cam = load_cam(path, FLAGS.interval_scale)  # 返回 np.array, shape=[2,4,4]
#             return cam.astype(np.float32)
#
#         cam = tf.py_function(func=load_cam_np, inp=[cam_path], Tout=tf.float32)
#         cam.set_shape([2, 4, 4])  # 必须固定 shape，否则 TF1.x map 会报错
#
#         return img, cam
#
#     dataset = tf.data.Dataset.from_tensor_slices((image_paths, cam_paths))
#     dataset = dataset.map(load_sample, num_parallel_calls=4)  # TF1.x 多线程异步
#     dataset = dataset.batch(FLAGS.batch_size)
#     dataset = dataset.prefetch(4)
#
#     # data from dataset via iterator
#     mvs_iterator = mvs_set.make_initializable_iterator()
#     scaled_images, centered_images, scaled_cams, image_index = mvs_iterator.get_next()
#
#     # set shapes
#     scaled_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
#     centered_images.set_shape(tf.TensorShape([None, FLAGS.view_num, None, None, 3]))
#     scaled_cams.set_shape(tf.TensorShape([None, FLAGS.view_num, 2, 4, 4]))
#     depth_start = tf.reshape(
#         tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
#     depth_interval = tf.reshape(
#         tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
#     depth_num = tf.cast(
#         tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []), 'int32')
#
#     # deal with inverse depth
#     if FLAGS.regularization == '3DCNNs' and FLAGS.inverse_depth:
#         depth_end = tf.reshape(
#             tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]), [FLAGS.batch_size])
#     else:
#         depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval
#
#     # depth map inference using 3DCNNs
#     if FLAGS.regularization == '3DCNNs':
#         init_depth_map, prob_map = inference_mem(
#             centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval)
#
#         if FLAGS.refinement:
#             ref_image = tf.squeeze(tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1)
#             refined_depth_map = depth_refine(
#                 init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, True)
#
#     # depth map inference using GRU
#     elif FLAGS.regularization == 'GRU':
#         init_depth_map, prob_map = inference_winner_take_all(centered_images, scaled_cams,
#             depth_num, depth_start, depth_end, reg_type='GRU', inverse_depth=FLAGS.inverse_depth)
#
#     # init option
#     init_op = tf.global_variables_initializer()
#     var_init_op = tf.local_variables_initializer()
#
#     # GPU grows incrementally
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#
#     with tf.Session(config=config) as sess:
#
#         # initialization
#         sess.run(var_init_op)
#         sess.run(init_op)
#         total_step = 0
#
#         # load model
#         if FLAGS.pretrained_model_ckpt_path is not None:
#             restorer = tf.train.Saver(tf.global_variables())
#             restorer.restore(
#                 sess, '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)]))
#             print(Notify.INFO, 'Pre-trained model restored from %s' %
#                   ('-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])), Notify.ENDC)
#             total_step = FLAGS.ckpt_step
#
#         # run inference for each reference view
#         sess.run(mvs_iterator.initializer)
#         for step in range(len(mvs_list)):
#
#             start_time = time.time()
#             try:
#                 out_init_depth_map, out_prob_map, out_images, out_cams, out_index = sess.run(
#                     [init_depth_map, prob_map, scaled_images, scaled_cams, image_index])
#             except tf.errors.OutOfRangeError:
#                 print("all dense finished")  # ==> "End of dataset"
#                 break
#             duration = time.time() - start_time
#             print(Notify.INFO, 'depth inference %d finished. (%.3f sec/step)' % (step, duration),
#                   Notify.ENDC)
#
#             # squeeze output
#             out_init_depth_image = np.squeeze(out_init_depth_map)
#             out_prob_map = np.squeeze(out_prob_map)
#             out_ref_image = np.squeeze(out_images)
#             out_ref_image = np.squeeze(out_ref_image[0, :, :, :])
#             out_ref_cam = np.squeeze(out_cams)
#             out_ref_cam = np.squeeze(out_ref_cam[0, :, :, :])
#             out_index = np.squeeze(out_index)
#
#             # paths
#             init_depth_map_path = output_folder + ('/%08d_init.pfm' % out_index)
#             prob_map_path = output_folder + ('/%08d_prob.pfm' % out_index)
#             out_ref_image_path = output_folder + ('/%08d.jpg' % out_index)
#             out_ref_cam_path = output_folder + ('/%08d.txt' % out_index)
#
#             # save output
#             write_pfm(init_depth_map_path, out_init_depth_image)
#             write_pfm(prob_map_path, out_prob_map)
#             out_ref_image = cv2.cvtColor(out_ref_image, cv2.COLOR_RGB2BGR)
#             image_file = file_io.FileIO(out_ref_image_path, mode='w')
#             scipy.misc.imsave(image_file, out_ref_image)
#             write_cam(out_ref_cam_path, out_ref_cam)
#             total_step += 1

def build_mvs_list_from_pairs(dense_folder, view_num):
    img_dir = os.path.join(dense_folder, "images_mvsnet")
    cam_dir = os.path.join(dense_folder, "cams")
    pair_path = os.path.join(dense_folder, "pair.txt")
    assert os.path.isdir(img_dir), f"missing dir: {img_dir}"
    assert os.path.isdir(cam_dir), f"missing dir: {cam_dir}"
    assert os.path.isfile(pair_path), f"missing file: {pair_path}"

    def id2paths(i):
        name = f"{int(i):08d}"
        return (os.path.join(img_dir, name + ".jpg"),
                os.path.join(cam_dir, name + "_cam.txt"))

    with open(pair_path, "r") as f:
        lines = [x.strip() for x in f.readlines()]
    n = int(lines[0]); p = 1

    mvs_list = []
    for _ in range(n):
        ref_id = int(lines[p]); p += 1
        parts = lines[p].split(); p += 1
        k = int(parts[0])
        # pair.txt 行：k  j1 s1  j2 s2 ...   取奇数位是 j*
        src_ids = [int(parts[i]) for i in range(1, 2*min(k, view_num-1)+1, 2)]

        ref_img, ref_cam = id2paths(ref_id)
        sample = [ref_img, ref_cam]
        for sid in src_ids[:view_num-1]:
            im, cm = id2paths(sid)
            sample += [im, cm]
        # 不足用 ref 填充
        while len(sample) < view_num * 2:
            sample += [ref_img, ref_cam]

        # 仅保留“文件都在”的样本
        ok = True
        for t in range(0, view_num*2, 2):
            if not (os.path.isfile(sample[t]) and os.path.isfile(sample[t+1])):
                ok = False; break
        if ok:
            mvs_list.append(sample)
    return mvs_list

def mvsnet_pipeline(mvs_list):
    """
    官方原版风格的推理流程（已修复：
    - mvs_list 数据格式
    - 生成器输出 shape
    - 稳健加载 ckpt（仅恢复匹配变量）
    - 会话 config 作用域
    - 关键日志
    ）
    """
    print('Testing sample number:', len(mvs_list))

    # --- 输出目录 ---
    output_folder = os.path.join(FLAGS.dense_folder, 'depths_mvsnet')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # --- Dataset from generator ---
    mvs_generator = iter(MVSGenerator(mvs_list, FLAGS.view_num))
    generator_data_type = (tf.float32, tf.float32, tf.float32, tf.int32)
    mvs_set = tf.data.Dataset.from_generator(lambda: mvs_generator, generator_data_type)
    mvs_set = mvs_set.batch(FLAGS.batch_size)
    mvs_iterator = mvs_set.make_initializable_iterator()
    scaled_images, centered_images, scaled_cams, image_index = mvs_iterator.get_next()

    # --- 固定 shape，避免 TF1.x 推断失败 ---
    scaled_images.set_shape([None, FLAGS.view_num, None, None, 3])
    centered_images.set_shape([None, FLAGS.view_num, None, None, 3])
    scaled_cams.set_shape([None, FLAGS.view_num, 2, 4, 4])

    # --- depth 参数 ---
    depth_start = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 0], [FLAGS.batch_size, 1, 1, 1, 1]),
        [FLAGS.batch_size]
    )
    depth_interval = tf.reshape(
        tf.slice(scaled_cams, [0, 0, 1, 3, 1], [FLAGS.batch_size, 1, 1, 1, 1]),
        [FLAGS.batch_size]
    )
    depth_num = tf.cast(
        tf.reshape(tf.slice(scaled_cams, [0, 0, 1, 3, 2], [1, 1, 1, 1, 1]), []),
        'int32'
    )
    if FLAGS.regularization == '3DCNNs' and FLAGS.inverse_depth:
        depth_end = tf.reshape(
            tf.slice(scaled_cams, [0, 0, 1, 3, 3], [FLAGS.batch_size, 1, 1, 1, 1]),
            [FLAGS.batch_size]
        )
    else:
        depth_end = depth_start + (tf.cast(depth_num, tf.float32) - 1) * depth_interval

    # --- 网络 ---
    if FLAGS.regularization == '3DCNNs':
        init_depth_map, prob_map = inference_mem(
            centered_images, scaled_cams, FLAGS.max_d, depth_start, depth_interval
        )
        if FLAGS.refinement:
            ref_image = tf.squeeze(
                tf.slice(centered_images, [0, 0, 0, 0, 0], [-1, 1, -1, -1, 3]), axis=1
            )
            refined_depth_map = depth_refine(
                init_depth_map, ref_image, FLAGS.max_d, depth_start, depth_interval, True
            )
    elif FLAGS.regularization == 'GRU':
        init_depth_map, prob_map = inference_winner_take_all(
            centered_images, scaled_cams,
            depth_num, depth_start, depth_end,
            reg_type='GRU', inverse_depth=FLAGS.inverse_depth
        )
    else:
        raise ValueError("Unknown regularization: " + str(FLAGS.regularization))

    # --- init & config（注意：config 必须在同一作用域里定义） ---
    init_op = tf.global_variables_initializer()
    var_init_op = tf.local_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = 0
    config.inter_op_parallelism_threads = 0

    # --- Session ---
    with tf.Session(config=config) as sess:
        # 初始化
        sess.run(var_init_op)
        sess.run(init_op)

        # 稳健 restore（只恢复匹配变量，自动跳过 LayerNorm 残留）
        if FLAGS.pretrained_model_ckpt_path is not None:
            ckpt_prefix = '-'.join([FLAGS.pretrained_model_ckpt_path, str(FLAGS.ckpt_step)])
            reader = tf.train.NewCheckpointReader(ckpt_prefix)
            ckpt_vars = reader.get_variable_to_shape_map()

            to_restore, to_skip = [], []
            for v in tf.global_variables():
                name = v.name.split(':')[0]
                if name in ckpt_vars and list(v.shape) == list(ckpt_vars[name]):
                    to_restore.append(v)
                else:
                    to_skip.append(name)

            print("[RESTORE] will load {} vars, skip {} vars".format(len(to_restore), len(to_skip)))
            for n in to_skip:
                if "LayerNorm" in n or "layer_norm" in n:
                    print("[SKIP-LN]", n)

            # 再次全量 init，随后 restore 覆盖匹配项
            sess.run(tf.global_variables_initializer())
            restorer = tf.train.Saver(var_list=to_restore)
            restorer.restore(sess, ckpt_prefix)
            print("[RESTORE] done. Pre-trained model restored from %s" % ckpt_prefix)

        print("[INFO] samples:", len(mvs_list), "view_num:", FLAGS.view_num, "batch:", FLAGS.batch_size)
        assert len(mvs_list) > 0, "mvs_list is empty. Check your dense_folder structure."

        # 迭代器初始化 & 推理 loop
        sess.run(mvs_iterator.initializer)
        print("[INFO] iterator initialized, start inference loop...")

        for step in range(len(mvs_list)):
            start_time = time.time()
            try:
                out_init_depth_map, out_prob_map, out_images, out_cams, out_index = sess.run(
                    [init_depth_map, prob_map, scaled_images, scaled_cams, image_index]
                )
            except tf.errors.OutOfRangeError:
                print("All dense finished")
                break

            duration = time.time() - start_time
            print('depth inference %d finished. (%.3f sec/step)' % (step, duration))

            # 保存
            for b in range(out_images.shape[0]):
                idx = int(np.squeeze(out_index[b]))
                init_depth_map_path = os.path.join(output_folder, '%08d_init.pfm' % idx)
                prob_map_path = os.path.join(output_folder, '%08d_prob.pfm' % idx)
                out_ref_image_path = os.path.join(output_folder, '%08d.jpg' % idx)
                out_ref_cam_path = os.path.join(output_folder, '%08d.txt' % idx)

                write_pfm(init_depth_map_path, np.squeeze(out_init_depth_map[b]))
                write_pfm(prob_map_path, np.squeeze(out_prob_map[b]))
                img_rgb = np.squeeze(out_images[b, 0])                     # (H,W,3) RGB
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_ref_image_path, img_bgr)
                write_cam(out_ref_cam_path, np.squeeze(out_cams[b, 0]))


def build_mvs_list(image_dir, cam_dir, view_num):
    """
    构造官方格式 mvs_list
    每条数据格式：[ref_img, ref_cam, src1_img, src1_cam, src2_img, src2_cam, ...]
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    mvs_list = []

    for idx, ref_img_file in enumerate(image_files):
        ref_img_path = os.path.join(image_dir, ref_img_file)
        ref_cam_path = os.path.join(cam_dir, ref_img_file.replace('.jpg', '_cam.txt'))

        if not os.path.exists(ref_cam_path):
            continue

        # 构造 source views（挑 view_num-1 个邻近视图）
        src_data = []
        count = 0
        for offset in range(1, len(image_files)):
            # 左右选择邻近图
            src_idx = idx - offset
            if src_idx < 0:
                src_idx = idx + offset
                if src_idx >= len(image_files):
                    break
            src_img_file = image_files[src_idx]
            src_img_path = os.path.join(image_dir, src_img_file)
            src_cam_path = os.path.join(cam_dir, src_img_file.replace('.jpg', '_cam.txt'))
            if not os.path.exists(src_cam_path):
                continue
            src_data.extend([src_img_path, src_cam_path])
            count += 1
            if count >= view_num - 1:
                break

        # 完整条目
        entry = [ref_img_path, ref_cam_path] + src_data
        mvs_list.append(entry)

    return mvs_list


def main(_):
    print('Testing MVSNet with totally %d view inputs (including reference view)' % FLAGS.view_num)

    mvs_list = build_mvs_list_from_pairs(FLAGS.dense_folder, FLAGS.view_num)

    print("mvs_list length:", len(mvs_list))
    if len(mvs_list) > 0:
        print("first entry (truncated):", mvs_list[0][:min(2*FLAGS.view_num, 6)])
    else:
        print("❌ mvs_list is EMPTY. Check images_mvsnet/, cams/, pair.txt under:", FLAGS.dense_folder)
        return

    mvsnet_pipeline(mvs_list)




if __name__ == '__main__':
    tf.app.run(main)
