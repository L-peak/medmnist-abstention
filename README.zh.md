# 医学影像拒识基准（Selective Prediction Benchmark）

选题：**When should a medical image classifier abstain?** —— 跨模态、跨来源漂移下，
8 种拒识分数（MSP / Entropy / 温度缩放 / Margin / MC Dropout / Deep Ensemble /
Conformal LAC / Conformal APS）能否守住风险预算，以及“在验证集上定阈值”这种做法本身是否成立。

数据用 MedMNIST v2（12 个 2D 子集，CC BY 4.0，`pip install medmnist` 直接下载），
关键结论再用 MedMNIST+ 224×224 复现。

## 目录结构

```
medmnist-abstention/
  src/config.py        实验配置：数据集清单、方法清单、路径
  src/data.py          MedMNIST 加载 + 合成数据回退（无需网络/依赖即可自测）
  src/models.py        ResNet-18（MedMNIST 适配）+ SmallCNN（CPU 快跑）
  src/train.py         训练/评估（AMP、checkpoint）
  src/abstention.py    八种拒识分数（conformal 两种都按预测集大小打分）
  src/metrics.py       risk-coverage / AURC / AUGRC / ECE / 验证集阈值选择（纯 NumPy）
  scripts/run_baseline.py   训练一个子集并保存 checkpoint
  scripts/run_selective.py  载入 logit bundle，跑全部拒识分数并导出结果
  scripts/report_verdicts.py 每个子集的胜者、并列、配对 bootstrap 置信区间
  scripts/make_latex_tables.py 由结果 CSV 生成论文表格 paper/tables/*.tex
  scripts/paper_numbers.py  汇总论文正文引用的全部数字 -> paper/NUMBERS.md
  scripts/smoke_test.py     合成数据端到端自测（不需要 medmnist、不需要 GPU）
  scripts/verify_paper_claims.py  用第二套独立实现重算论文的核心结论并断言（80 个 operating point 等）
  scripts/exchangeability_control.py  对照实验：测试集内部二分做校准/评估（同分布），验证协议本身没问题
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 先验管线（合成数据，秒级，不联网）
python scripts/smoke_test.py

# 2) 真实数据：训练并导出 logits（这一步才需要 GPU）
python scripts/run_baseline.py --dataset pathmnist --size 28 --epochs 20 --seed 0
python scripts/run_baseline.py --dataset pathmnist --size 28 --epochs 20 --seed 1   # 集成用

# 3) 拒识方法对比（CPU 即可，只读 logits）
python scripts/run_selective.py --stem pathmnist_resnet18_28

# 4) 出图
python scripts/plot_risk_coverage.py --npz outputs/results/pathmnist_resnet18_28_risk_coverage.npz
```

## GPU / CPU 分工（Kaggle T4 x2 场景）

`run_baseline.py` 训练结束后会把 val/test 的 logits、标签、MC-Dropout 概率
打包成 `outputs/logits/<stem>_seed<k>_logits.npz`（几十 MB）。把这个文件下载回来，
`run_selective.py` 和 `plot_risk_coverage.py` 在本机 CPU 上就能跑完全部分析——
GPU 只负责训练，不浪费 Kaggle 每周 30 小时的配额。

注意：**不要把 torch 和 matplotlib 放进同一个进程**（Windows/conda 会因
libiomp5md.dll 重复加载直接崩），所以画图是独立脚本、独立进程。

另一个实测坑：**`num_workers > 0` 在本机会慢 60 倍**（2 个 epoch：1.3 秒 vs 79.7 秒）。
原因是 Windows 用 spawn 启进程，每个 epoch 都会重新拉起 worker 并重新导入 torch。
代码里已经加了 `persistent_workers=True` 兜底，默认值也改成 0——MedMNIST 数据全在
内存里，worker 本来也帮不上什么忙。

## Kaggle 上跑整批实验

```bash
# 一个 cell 跑完整个网格：可断点续跑、到点自动收工、最后打包成一个 zip
!python scripts/kaggle_run_all.py --max-minutes 480 --seeds 0,1,2 --zip

# 只想看看计划和哪些已经跑过
!python scripts/kaggle_run_all.py --dry-run
```

会话中断后重新执行同一条命令即可：已有 bundle 的 run 会被跳过（实测 0.0 分钟全部跳过）。
结束时把 `outputs/results/logit_bundles.zip` 下载回来，在本机跑：

```bash
python scripts/run_selective.py --stem pathmnist_resnet18_28
python scripts/plot_risk_coverage.py --npz outputs/results/pathmnist_resnet18_28_risk_coverage.npz
```

## GPU 说明（T4 ×2）

Turing 架构（T4）**不支持 bf16 和 TF32**，所以训练脚本在 CUDA 下用
`torch.amp.autocast(dtype=torch.float16)` + `GradScaler`，不要用 `bfloat16`。
28×28 的子集 CPU 就能跑完；224 版本的 MedMNIST+ 用单张 T4 足够，双卡用 DDP
（无 NVLink，只能数据并行）。

## 实验矩阵

| 组 | 内容 | 产物 |
| --- | --- | --- |
| E1 | 12 个子集 × 8 种拒识分数（3 个种子） | AURC / AUGRC / post-hoc coverage@risk 表 |
| E2 | 验证集定阈值 → 冻结 → 测试集评估 | 可部署 operating point、风险预算是否守住 |
| E3 | 5 个子集 128×128 复验（2 个种子） | 分辨率对分数排名的影响 |
| E4 | 二分类退化、多标签风险口径、无效区分析 | 何时这些比较本身没有意义 |

## 指标口径（写论文时按这套描述）

- `risk = 1 - accuracy`（多标签为 Hamming error），按置信度降序累积 → risk-coverage 曲线
- `AURC`：曲线下面积，越低越好；`E-AURC`：相对 oracle 排序的超出量
- `coverage@risk<=r`：**post-hoc**，直接用测试曲线选点，只用于方法比较，不是部署保证
- `coverage_val_selected` / `risk_val_selected`：阈值在验证集选定后冻结，测到的真实测试风险
- `AUGRC`：替代 AURC 的新指标（Traub et al., NeurIPS 2024）
- `ECE`：置信度校准误差

## 已知结论（v1.1）

- Ensemble 在 12 个子集里 7 个严格最优、4 个并列、1 个（pneumoniamnist）名义落后；
  相对最优单模型的提升只有 3/12 个子集统计显著；128×128 无一显著。
- 两种 conformal（按预测集大小打分）在 8 个能区分分数的子集里全部垫底；
  LAC 的测试集实际覆盖率均值 0.816，低于名义 0.900（11 个子集中有 10 个低于名义）。
- 验证集定的阈值在测试集上 54/80 个超过 10% 风险预算，最高达验证集估计的 6.2 倍；
  原因是 MedMNIST 的验证/测试划分来自不同来源、类别先验漂移。
- 对照实验（把测试集二分，一半校准一半评估，两边同分布）：同一套规则平均风险 0.097（预算 0.10），
  LAC 覆盖率回到 0.901（名义 0.900）——说明失效来自"划分"而不是"定阈值的规则"。

## 下一步（未做）

- 224×224（MedMNIST+ 全套 27 GB，超过 Kaggle 免费额度 20 GB 磁盘）。
- 单一 ResNet-18 backbone；更强的 backbone 下 ensemble 的相对优势是否变化未做。
- 从训练集里再切一个独立校准集（不参与模型选择），用来验证论文给出的协议建议。
