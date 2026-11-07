"""绘制 SEU 指定工况五种状态的行星箱三轴原始振动。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

condition = '30_2'
setting = '30 Hz–2 V'

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '工作目录/outputs/SEU-数据直观认识'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [('Health', '健康'), ('Chipped', '齿部故障'), ('Miss', '缺齿'),
         ('Root', '齿根故障'), ('Surface', '齿面故障')]
plt.rcParams.update({'font.family': ['PingFang SC', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 11,
                     'agg.path.chunksize': 10000})
records = []
for name, label in CASES:
    path = ROOT / f'原始数据/SEU/gearbox/gearset/{name}_{condition}.csv'
    with path.open() as f:
        for _ in range(16):
            next(f)
        # 源文件用字面量反斜线 t 分隔；转换仅发生在读取流中。
        data = np.loadtxt((line.replace('\\t', '\t') for line in f), usecols=(1, 2, 3))
    assert data.ndim == 2 and data.shape[1] == 3 and np.isfinite(data).all()
    records.append((name, label, data))
limits = np.max([np.abs(x).max(axis=0) for _, _, x in records], axis=0) * 1.05
summary = []
for name, label, data in records:
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    indices = np.arange(len(data))
    for j, (axis, color) in enumerate(zip(axes, ['#24649b', '#147c73', '#b66328'])):
        axis.plot(indices, data[:, j], lw=.35, color=color)
        axis.set_title(f'行星齿轮箱 {"xyz"[j]} 方向振动 · 第 {j+2} 通道', loc='left', fontsize=12)
        axis.set_ylabel('振动信号幅值')
        axis.set_ylim(-limits[j], limits[j])
        axis.set_xlim(0, len(data)-1)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.ticklabel_format(axis='x', style='sci', scilimits=(6, 6), useMathText=True)
    axes[-1].set_xlabel('采样点序号（从 0 开始）')
    fig.suptitle(f'{label}（{name}）｜行星齿轮箱三轴振动', fontsize=20, x=.085, ha='left', y=.98)
    fig.text(.085, .927, f'{name}_{condition}.csv  ·  {setting} 设置  ·  每轴 {len(data):,} 个点  ·  对应方向在五张图中统一纵轴范围', color='#505b66')
    fig.text(.085, .022, '完整原始记录，未滤波、归一化或截段；横轴使用采样点序号。', color='#505b66')
    fig.tight_layout(rect=(.025, .05, .99, .90), h_pad=1.8)
    target = OUT / f'{name}_{condition}_three_axes.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    summary.append({'file': f'{name}_{condition}.csv', 'points_per_axis': len(data),
                    'channels_one_based': [2, 3, 4], 'minimum': data.min(axis=0).tolist(),
                    'maximum': data.max(axis=0).tolist(), 'figure': target.name})
    print(target, flush=True)
(OUT / f'绘图清单_{condition}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
