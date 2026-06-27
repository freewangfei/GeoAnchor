# GeoAnchor-Net 中文说明

[English README](README.md)

GeoAnchor-Net 是一个面向地震驱动孔隙度预测的监督感知模型，可用于密集标签全场重建和稀疏井盲区间预测。模型在同一训练与推理流程中结合地质趋势构建、结构条件残差分配和绝对孔隙度锚点传播。

本仓库仅包含 GeoAnchor-Net 的训练与评价代码，不包含数据集下载脚本、辅助分析脚本、基线模型或早期内部方法。

## 网络概览

![GeoAnchor-Net architecture](fig/fig1.png)

图 1 展示 GeoAnchor-Net 的整体流程。模型输入包括地震数据、结构属性、可选先验场、稀疏孔隙度锚点和锚点掩膜。趋势分支首先构建稳定的背景孔隙度场，残差分支随后预测结构条件修正分量，锚点分支将稀疏井观测作为绝对孔隙度值传播，并完成最终的锚点一致融合。

![GeoAnchor-Net modules](fig/fig2.png)

图 2 展示最终模型使用的两个主要模块。结构条件加权分支将残差分量分配到不同地质上下文，例如平滑区域、结构过渡区和井支撑区。锚点引导分支保持稀疏井样本与目标孔隙度处于同一物理变量尺度，避免仅将井观测视为相对于先验的残差扰动。

## 实验示例

![F3 blind-interval curve comparison A](fig/fig6a.png)
![F3 blind-interval curve comparison B](fig/fig6b.png)

图 6 展示 F3 Demo 2023 上具有代表性的盲区间井曲线对比。GeoAnchor-Net 与基于先验和学习型替代方法在保留深度区间上进行比较。曲线图用于观察稀疏锚点信息是否改善局部孔隙度轨迹，而不仅仅降低图像级误差。

![F3 local-window analysis A](fig/fig7a.png)
![F3 local-window analysis B](fig/fig7b.png)

图 7 展示 F3 上的局部窗口分析。各面板比较真实值、Prior-Only 预测、GeoAnchor-Net 预测、评价掩膜和局部误差降低。误差降低图中的暖色区域表示 GeoAnchor-Net 相对于 Prior-Only 降低局部误差的位置。

![F3 crossplot](fig/fig8.png)

图 8 展示 F3 评价点上预测孔隙度与参考孔隙度的交会图。点云越接近恒等线，表示与保留盲区间样本的一致性越好。

## 仓库内容

```text
GeoAnchor_code/
  geoanchor/
    data.py          # NPZ 数据集加载器
    metrics.py       # 回归指标
    model.py         # GeoAnchor-Net 模型
    train_eval.py    # 训练、验证集校准和测试
  fig/
    fig1.pdf
    fig1.png
    fig2.pdf
    fig2.png
    fig6a.pdf
    fig6a.png
    fig6b.pdf
    fig6b.png
    fig7a.pdf
    fig7a.png
    fig7b.pdf
    fig7b.png
    fig8.pdf
    fig8.png
  train.py           # 单个数据集训练入口
  test.py            # 单个 checkpoint 评价入口
  run_reproduce.py   # 复现四组 GeoAnchor-Net 实验
  requirements.txt
```

## 数据

请将数据集放在 `GeoAnchor_code/data/` 下，并保持如下结构：

```text
data/
  openporobench_s/
    train.npz
    val.npz
    test.npz
  external/
    f3_demo_2023/
      openporo_npz/
        train.npz
        val.npz
        test.npz
    seis2rock_dense_openporo/
      train.npz
      val.npz
      test.npz
    seis2rock_aux_openporo/
      train.npz
      val.npz
      test.npz
```

数据来源：

- F3 Demo 2023: https://terranubis.com/datainfo/F3-Demo-2023
- Seis2Rock-Smeaheia: https://zenodo.org/records/11481946
- OpenPoroBench-S: 随本仓库生成并发布的合成划分

每个 `.npz` 划分应包含：

- `seismic`: 形状 `(N, 1, H, W)`
- `structure`: 形状 `(N, C, H, W)`
- `porosity`: 形状 `(N, 1, H, W)`
- `domains`: 形状 `(N,)`

可选字段：

- `prior`: 形状 `(N, 1, H, W)`
- `supervised_mask`: 已观测稀疏井样本，形状 `(N, 1, H, W)`
- `eval_mask`: 保留评价样本，形状 `(N, 1, H, W)`

如果缺少 `prior`，代码会使用 `structure[:, 0]` 从训练划分构建一个简单的 RGT 到孔隙度趋势先验。

## 环境

```bash
conda activate your_env
cd GeoAnchor_code
pip install -r requirements.txt
```

脚本会在 CUDA 可用时使用 GPU。若没有 CUDA，可使用 `--device cpu`。GPU 运行时请安装与本地 CUDA 驱动匹配的 PyTorch 版本。代码已在 `torch==2.11.0+cu128`、`numpy==1.26.4` 和 `scikit-image==0.26.0` 下检查。

## 训练

OpenPoroBench-S:

```bash
python train.py \
  --data-root data/openporobench_s \
  --out-dir outputs/openporobench_s \
  --epochs 44 \
  --batch-size 8 \
  --lr 1.2e-4 \
  --seed 20260446 \
  --calibration-mode quadratic_prior \
  --disable-prior-condition \
  --anchor-band-strength 0.65 \
  --curve-gate-strength 0.65
```

F3 Demo 2023:

```bash
python train.py \
  --data-root data/external/f3_demo_2023/openporo_npz \
  --out-dir outputs/f3_demo_2023 \
  --epochs 20 \
  --batch-size 2 \
  --lr 7e-4 \
  --seed 20260431 \
  --calibration-mode none
```

Seis2Rock dense split:

```bash
python train.py \
  --data-root data/external/seis2rock_dense_openporo \
  --out-dir outputs/seis2rock_dense \
  --epochs 26 \
  --batch-size 4 \
  --lr 3e-4 \
  --seed 20260547 \
  --calibration-mode cubic_prior_relative \
  --disable-prior-condition \
  --anchor-band-strength 0.65 \
  --curve-gate-strength 0.65
```

Seis2Rock auxiliary split:

```bash
python train.py \
  --data-root data/external/seis2rock_aux_openporo \
  --out-dir outputs/seis2rock_aux \
  --epochs 24 \
  --batch-size 4 \
  --lr 2e-4 \
  --seed 20260727 \
  --calibration-mode cubic_prior_relative
```

## 测试

```bash
python test.py \
  --checkpoint outputs/f3_demo_2023/geoanchor_net.pt \
  --data-root data/external/f3_demo_2023/openporo_npz \
  --out-dir outputs/f3_demo_2023_test
```

测试输出包括：

- `test_metrics.json`
- `test_predictions.npz`

## 复现论文报告的 GeoAnchor-Net 实验

```bash
python run_reproduce.py
```

汇总结果会写入：

```text
outputs/reproduction_summary.json
```
