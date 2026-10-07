# CatNet-Test-Pytorch

CatNet 的 PyTorch MobileNet 分类实验，用于识别猫的 7 种毛色／花纹。包含训练、单图预测、猫脸裁剪辅助代码、已保存的 PTH 权重和 ONNX 模型。

主应用与所有模块入口：[CatNet-Unity](https://github.com/Pannic17/CatNet-Unity)。TensorFlow 主训练实验见 [CatNet-Tensorflow](https://github.com/Pannic17/CatNet-Tensorflow)，数据裁剪见 [CatNet-Face-Cut](https://github.com/Pannic17/CatNet-Face-Cut)。

## 文件说明

| 文件 | 用途 |
| --- | --- |
| `model_v2.py`、`model_v3.py` | PyTorch MobileNetV2 / MobileNetV3 定义 |
| `train.py` | 7 类 MobileNetV2 训练、验证、PTH 保存和 ONNX 导出 |
| `predict.py` | 使用 `MobileNetV2_Cat.pth` 的单图预测 |
| `test.py`、`detection.py` | 猫脸裁剪后预测实验和 OpenCV 辅助函数 |
| `split_data.py` | 从 `cat_data/cat_face` 划分 train / val |
| `class_indices.json` | 7 类标签映射 |
| `*.pth`、`*.onnx` | 不同实验的权重和导出模型 |
| `train_mobilenet_v2.py`、`trainGPU_mobilenet_v2.py`、`train_mobilenet_v3.py`、`utils.py`、`read_ckpt.py`、`trans_v3_weights.py` | 保留的 TensorFlow 实验代码，不能作为现有 PyTorch 网络的直接训练入口 |

类别顺序为 `Bicolor`、`Calico`、`Colorpoint`、`Mix`、`Orange`、`Solid`、`Tabby`，表示外观类别而非品种。

## 环境

PyTorch 主流程使用 Python、torch、torchvision、Pillow、Matplotlib、tqdm；猫脸实验额外使用 OpenCV，ONNX 导出使用 onnx。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch torchvision pillow matplotlib tqdm opencv-python onnx
```

依赖版本没有锁定。训练代码在验证阶段直接调用 `.cuda()`，导出输入也固定为 `device='cuda'`，因此原样训练需要 CUDA 可用的 PyTorch 与 GPU；以上基础安装命令不保证 CUDA 配置完成。CPU 训练需要先调整这些位置。预测入口会按 CUDA 可用性选择 GPU 或 CPU。

## 数据与训练

`train.py` 从仓库当前工作目录读取以下结构；完整数据集未提供：

```text
data_set/cat_data/
  train/<类别名>/*.jpg
  val/<类别名>/*.jpg
```

两组目录都应有相同的 7 类。`split_data.py` 使用的是另一个根路径 `cat_data/`；若要使用该工具，请先将其路径改成 `data_set/cat_data`，或将划分结果放到训练入口期望的位置。工具按 90% / 10% 划分，**会先删除已有 train / val 目录再重建**。

配置好数据和 CUDA 后，从仓库根目录执行：

```powershell
python train.py
```

默认 batch size 为 16，训练 24 轮，Adam 学习率为 0.0001，DataLoader worker 为 8。预训练加载代码被注释，默认从随机初始化训练。验证准确率提升时保存 `CMN_b16e24_pytorch_v2.pth` 和 `CNM_b16e24_pytorch_v2.onnx`，并重写类别映射文件。

## 预测

修改 `predict.py` 的 `img_path`，将原作者 `H:` 盘图片目录改成自己的目录。默认权重 `MobileNetV2_Cat.pth` 已在仓库内；使用新训练结果时改为相应 PTH 路径，并保证标签映射匹配。

```powershell
python predict.py
```

按提示输入包含扩展名的文件名。预测将图片 Resize 至 256、中心裁剪至 224，转换为 NCHW Tensor，并使用 ImageNet 均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]`。对模型 logits 应用 `torch.softmax`，显示类别与概率。

`test.py` 还需要修改测试图像目录与 `detection.py` 中的级联 XML 路径，XML 可从 Face-Cut 仓库取得。

## ONNX 与 Unity 接口差异

`train.py` 导出 opset 11、输入 `(1, 3, 224, 224)`、输入名 `input`、输出名 `softmax`。**输出名称不代表执行了 Softmax**：`model_v2.py` 返回分类 logits，预测脚本才显式计算 Softmax。

Unity 主仓库使用 RGB NHWC 与 `[-1, 1]` 归一化，这里使用 NCHW 和 ImageNet 标准化。接入前必须调整输入和输出处理并验证类别顺序，不能只替换模型文件。

仓库中的历史 TensorFlow 脚本与本地 PyTorch `model_v2.py` / `model_v3.py` 存在接口不匹配，需单独整理后使用。本文依据源码，未执行训练、推理或模型导出。
