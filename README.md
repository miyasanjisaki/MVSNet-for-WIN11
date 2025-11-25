# MVSNet-for-WIN11
It's for my final project.

# COLMAP → MVSNet 预处理脚本

本仓库包含一个用于将 **COLMAP 稀疏重建结果** 转换为 **MVSNet 所需数据格式** 的预处理脚本 `colmap2mvsnet.py`。  

脚本会完成以下工作：

- 从 COLMAP 导出的 `cameras.txt / images.txt / points3D.txt` 中读取相机和点云信息  
- 构建每张图片对应的 **内参矩阵 (K)** 与 **外参矩阵 ([R|t])**  
- 根据可见 3D 点的深度分布，自动估计每张图片的 **深度范围与采样间隔**  
- 通过几何一致性打分，自动进行 **视角选择 (view selection)**，生成 `pair.txt`  
- 将原始图像重命名为 `%08d.jpg`，与 `cams/`、`pair.txt` 一起构成标准的 **MVSNet 输入格式**

---

## 1. 目录结构约定

脚本假设数据组织如下（`--dense_folder` 指向的目录）：
```

dense_folder/
├── images/          # COLMAP 使用的原始图像
├── sparse/          # COLMAP 导出的稀疏模型（文本格式）
│   ├── cameras.txt
│   ├── images.txt
│   └── points3D.txt
└── （运行脚本后生成）
    ├── cams/        # 每张图像对应的相机参数和深度范围
    ├── pair.txt     # 视角配对文件（MVSNet 使用）
    └── images/      # 以 %08d.jpg 命名的图像（可能覆盖/追加到原目录）
```


当前脚本默认从 `sparse/` 目录下读取 **文本格式** 的 COLMAP 模型 (`.txt`)。如果你现在是 `.bin` 文件，需要先用 COLMAP 导出为 `.txt`。

---

## 2. 环境依赖

* Python 3.x
* 依赖库：

  * `numpy`
  * `opencv-python`
  * `multiprocessing`（标准库）
  * `argparse`、`os`、`shutil`、`struct` 等（标准库）

安装额外依赖示例：

pip install numpy opencv-python


---

## 3. 脚本功能概述

### 3.1 读取 COLMAP 模型

脚本包含一套 `read_model` 工具函数，用于读取 COLMAP 的稀疏模型（文本版）：

* `cameras.txt`

  * 解析相机 ID、模型类型（`SIMPLE_PINHOLE / PINHOLE / OPENCV / RADIAL` 等）
  * 解析相机参数（`fx, fy, cx, cy, k1, k2, ...`）

* `images.txt`

  * 解析每张图像的四元数 `qvec`、平移向量 `tvec`
  * 解析图像文件名、可见的 3D 点 ID 列表 `point3D_ids`

* `points3D.txt`

  * 解析每个 3D 点的三维坐标 `xyz` 以及被哪些图像观测到

### 3.2 构建相机内参 (intrinsic)

根据 COLMAP 的相机模型，将参数映射为 3×3 内参矩阵：

```
[ fx   0  cx ]
[  0  fy  cy ]
[  0   0   1 ]
```


对于只给了 `f` 的模型（如 `SIMPLE_PINHOLE`），脚本会自动令 `fx = fy = f`。

### 3.3 构建相机外参 (extrinsic)

对于每张图像：

* 将四元数 `qvec` 转换为旋转矩阵 `R`
* 和平移向量 `t` 组合成 4×4 外参矩阵：

```
[ R  t ]
[ 0  1 ]
```

### 3.4 深度范围与采样

对每一张图像：

1. 将该图像可见的 3D 点坐标变换到相机坐标系下
2. 取这些点的 z 值排序，区间 **[1%, 99%] 分位数** 作为 `depth_min` 和 `depth_max`
3. 如果 **未显式设置 `--max_d`**，使用逆深度策略自动计算深度层数 `depth_num`
4. 根据 `depth_min / depth_max / depth_num / interval_scale` 计算 `depth_interval`

最终得到一组参数：

depth_min  depth_interval  depth_num  depth_max


并写入 `cams/%08d_cam.txt` 文件。

### 3.5 视角选择 (view selection)

脚本对任意两张图像 (i, j)：

1. 找出它们共同观测到的 3D 点集合
2. 计算相机中心与这些点连线的夹角，得到基线角度 `theta`
3. 通过高斯权重函数（由 `theta0 / sigma1 / sigma2` 控制）累积得分
4. 使用多进程并行加速所有图像对的打分

然后对每一张图像 i：

* 按得分从高到低排序
* 选出前若干个参考视角，形成 `[(view_id, score), ...]`
* 写入 `pair.txt` 中（MVSNet 训练/推理会用到）

---

## 4. 使用方法

### 4.1 基本命令


python colmap2mvsnet.py \
    --dense_folder /path/to/your/dense_folder

示例：

```
python colmap2mvsnet.py \
    --dense_folder ./data/dam_scene
```


运行后，会在 `dense_folder` 下生成或更新：

* `cams/` 目录：每张图像的相机矩阵与深度范围
* `pair.txt`：视角配对关系
* `images/`：按 `%08d.jpg` 重命名的图像（可能与原 `images/` 目录相同）

 注意：脚本中 `image_dir` 与 `renamed_dir` 默认都为 `dense_folder/images`，也就是说 **重命名后的 `%08d.jpg` 会写回同一个目录**，建议：
* 先备份原始图像，或者在运行 COLMAP 时使用一个干净的 `images/` 目录，仅用于 MVSNet

---

### 4.2 参数说明

脚本支持以下命令行参数：

```
python colmap2mvsnet.py \
    --dense_folder DENSE_DIR \
    [--max_d MAX_D] \
    [--interval_scale INTERVAL] \
    [--theta0 THETA0] \
    [--sigma1 SIGMA1] \
    [--sigma2 SIGMA2] \
    [--test] \
    [--convert_format]
```

参数含义：

* `--dense_folder` (str, 必选)
  项目根目录，内含 `images/` 与 `sparse/`。

* `--max_d` (int, 默认 0)
  最大深度层数。

  * 为 0 时：按逆深度公式自动估计
  * 非 0 时：使用固定层数（例如 192）

* `--interval_scale` (float, 默认 1)
  深度间隔缩放因子。>1 会让深度采样更稀疏。

* `--theta0` (float, 默认 5)
  视角打分中区分“小基线”和“大基线”的角度阈值（单位：度）。

* `--sigma1` (float, 默认 1)、`--sigma2` (float, 默认 10)
  视角打分高斯权重的两个标准差，分别用于

  * `theta <= theta0`
  * `theta > theta0`

* `--test` (flag)
  若设置该参数，则 **只计算不写文件**，便于调试检查。

* `--convert_format` (flag)

  * 设置时：使用 OpenCV 读取原始图片并保存为 `%08d.jpg`（可用于格式转换，如 PNG → JPG）
  * 不设置时：直接用 `shutil.copyfile` 拷贝图像，并重命名为 `%08d.jpg`（不会改变文件格式）

---

## 5. 输出文件格式

### 5.1 `cams/%08d_cam.txt`

每张图像对应一个文件，例如 `00000000_cam.txt`，格式为：

```
extrinsic
r11 r12 r13 t1
r21 r22 r23 t2
r31 r32 r33 t3
0   0   0   1

intrinsic
fx  0  cx
0  fy  cy
0   0   1

depth_min depth_interval depth_num depth_max
```

* `extrinsic`：4×4 外参矩阵
* `intrinsic`：3×3 内参矩阵
* 最后一行为深度采样信息

### 5.2 `pair.txt`

`pair.txt` 用于描述每张图像的参考视角列表，典型格式如下：

```
N
0
10  1 s01  2 s02  ...
1
10  0 s10  2 s12  ...
...
```

* 首行 `N`：图像总数
* 后面对每张图像：

  * 一行：图像索引 i
  * 一行：`num_views` 后面跟着若干 `(view_id, score)` 对

MVSNet 会根据这些视角关系选择邻居视角进行多视图深度估计。

---

## 6. 调试信息

为了方便检查脚本是否正确读取了模型，运行时会输出若干调试信息，例如：

* 当前工作路径 `os.getcwd()`
* 模型目录 `model_dir` 与其中的文件列表
* `cameras` / `images` / `points3d` 的基本内容
* 示例：

  * `intrinsic[1]`
  * `extrinsic[1]`
  * `depth_ranges[1]`
  * `view_sel[0]`

如果发现重建效果异常，可以先确认这些矩阵和深度范围是否合理。

---

## 7. 典型使用流程（与三维重建项目结合）

在你的三维重建项目中，可以采用如下 pipeline：

1. **采集影像**：用无人机 / 相机拍摄视频或多张照片
2. **用脚本抽帧 / 预处理**（例如从视频中截取 400 张 640×512 的图像）
3. **使用 COLMAP**：

   * 特征提取 & 匹配
   * 稀疏重建，得到 `sparse/` 模型
4. **运行本脚本 `colmap2mvsnet.py`**：

   * 生成 `cams/`、`pair.txt`、重命名的 `images/`
5. **将输出输入到 MVSNet / R-MVSNet 等网络**：

   * 进行密集重建，获取高精度三维点云/深度图
6. （可选）导入到其它软件（如 Open3D / Meshlab / CloudCompare）进行后处理与可视化

---

## 8. 致谢

本脚本基于 HKUST **Jingyang Zhang** 和 **Yao Yao** 在 MVSNet 项目中提供的预处理代码，并在此基础上做了适配与调试输出增强，以便于在实际工程场景（例如水工建筑三维重建）中使用。

若你在科研或课程项目中使用了本脚本，建议在文中对 MVSNet 原作者及本预处理脚本来源进行适当致谢。

