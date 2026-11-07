"""用已计算的窗口特征绘制三轴 RMS—峭度散点图。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '工作目录/outputs/SEU-数据直观认识'
source = json.loads((OUT / '窗口特征_30_2.json').read_text())
records = source['records']
CASES = [('Health', '健康', '#2878b5', 'o'),
         ('Chipped', '齿部故障', '#e69f00', 's'),
         ('Miss', '缺齿', '#239b73', '^'),
         ('Root', '齿根故障', '#9867b4', 'D'),
         ('Surface', '齿面故障', '#d64b45', 'P')]
assert source['condition'] == '30_2' and len(records) == 1500
plt.rcParams.update({'font.family': ['PingFang SC', 'DejaVu Sans'],
                     'axes.unicode_minus': False, 'font.size': 11})
fig, axes = plt.subplots(1, 3, figsize=(18, 6.7), sharey=True)
for j, (ax, direction) in enumerate(zip(axes, 'xyz')):
    for state, label, color, marker in CASES:
        subset = sorted((r for r in records if r['state'] == state and r['axis'] == direction), key=lambda r:r['window'])
        assert [r['window'] for r in subset] == list(range(1,101))
        values = np.array([[r['rms'], r['kurtosis']] for r in subset])
        assert np.isfinite(values).all()
        ax.scatter(values[:,0], values[:,1], label=label, color=color, marker=marker,
                   s=32, alpha=.72, edgecolors='white', linewidths=.35)
    ax.set_title(f'{direction} 方向 · 第 {j+2} 通道', loc='left', fontsize=16, pad=14)
    ax.set_xlabel('RMS（振动信号幅值）', labelpad=10)
    ax.ticklabel_format(axis='x', style='plain', useOffset=False)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelleft=True)
    ax.margins(x=.08, y=.08)
axes[0].set_ylabel('峭度（无量纲）', labelpad=10)
fig.suptitle('30_2｜RMS 与峭度的联合分布', x=.055, ha='left', y=.97, fontsize=25)
fig.text(.055,.895,'每个点代表一个窗口 · 每种状态 100 个窗口 · 每窗 10,485 点 · 不重叠', fontsize=12, color='#52616c')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52,.845), ncol=5, frameon=False, fontsize=12)
fig.text(.055,.035,'三幅图横轴分别取范围，纵轴范围一致；展示全部窗口的原始特征值。峭度采用正态参考值为 3 的定义。',fontsize=11,color='#52616c')
fig.tight_layout(rect=(.015,.09,.995,.78),w_pad=2.5)
fig.savefig(OUT / 'RMS-峭度散点图_30_2.png', dpi=180)
plt.close(fig)
print('已绘制三个方向，每幅 500 个点，共 1,500 个点；每类窗口编号 1—100 完整。')
