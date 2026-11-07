# 随机森林训练

使用 Windows 主机 `win`，目录 `D:\SEU-training-20261104`。独立 Python 环境放在目录内的 `.venv`。

只上传前 80% 的 x、y、z 信号，在 Windows 计算 2048 点窗口的特征，执行 108 套配置的三折时间交叉验证，共 324 次训练。测试段没有读取或上传。

- 输入组合与参数：[模型配置](SEU-2048点配置记录.md)
- 本地数据准备：[prepare_seu_remote_data.py](../../scripts/prepare_seu_remote_data.py)
- 训练脚本：[train_seu_rf_cv.py](../../scripts/train_seu_rf_cv.py)
- Windows 入口：[run_seu_rf_windows.ps1](../../scripts/run_seu_rf_windows.ps1)

远端运行：`& D:\SEU-training-20261104\run_seu_rf_windows.ps1`。

进度在 `training.log`，结果在 `results`：保存每折 F1、混淆矩阵、验证预测、特征表、配置排名与环境版本。`requirements-lock.txt` 记录依赖版本。

当前运行只比较候选配置，最终模型训练与测试评价留到讨论结果后进行。

## 第一轮结果

Windows 已完成 108 套配置、324 次训练，耗时 166.5 秒。结果已取回，逐折核对混淆矩阵、宏平均 F1 和汇总均值，测试集未使用。

| 输入 | 最佳配置 | 三折宏平均 F1 | 平均值 |
| --- | --- | --- | --- |
| A | RF017 | 0.7988／0.7616／0.7819 | 0.7808 |
| B | RF065 | 0.8774／0.8290／0.8523 | 0.8529 |
| C | RF097 | 0.9409／0.9221／0.9166 | 0.9265 |

完整排名：[ranking.csv](../../outputs/SEU-RF训练/results/ranking.csv)；最佳参数与环境：[summary.json](../../outputs/SEU-RF训练/results/summary.json)、[environment.json](../../outputs/SEU-RF训练/results/environment.json)。同分配置按编号展示，具体选择接着讨论。

## 第二轮：扩大搜索

六组输入与更大的超参数网格，共 1620 套配置、4860 次训练，见 [模型配置](SEU-2048点配置记录.md)。第一轮资料保留。

远端目录：`D:\SEU-training-20261104\expanded-grid`。使用已有 Python 环境和前 80% 三轴信号，最多 8 个线程训练每个森林，配置依次执行。

每完成一套配置，保存三折成绩及预测到 `results/checkpoints`；中断后可跳过已完成项继续。进度见 `training.log`，依赖版本见 `requirements-lock.txt`。训练脚本：[train_seu_rf_expanded.py](../../scripts/train_seu_rf_expanded.py)。

第二轮已完成，耗时约 66.5 分钟。1620 套配置、4860 个折次的结果已取回并核对。C 组仍最高，三折宏 F1 均值 0.9272；测试段未使用。

六组最佳参数、误判对比与图片见 [扩大搜索结果](SEU-随机森林扩大搜索.md)。最终模型训练与测试评价留到讨论结果后进行。

## 正式训练完成

已按 C／EXP0721 配置，用前 80% 的 2040 个窗口训练，在后 20% 的 510 个测试窗口上评价：准确率 94.31%，宏 F1 0.9422。模型已保存，测试后未调参。上述封存状态对应选模阶段；最终结果见 [正式训练与测试](SEU-随机森林正式训练.md)。
