"""绘制跨方向 RMS 和同方向 RMS—峰值因子的联合分布。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '工作目录/outputs/SEU-数据直观认识'
source = json.loads((OUT / '窗口特征_30_2.json').read_text())
CASES = [('Health', '健康', '#2878b5', 'o'),
         ('Chipped', '齿部故障', '#e69f00', 's'),
         ('Miss', '缺齿', '#239b73', '^'),
         ('Root', '齿根故障', '#9867b4', 'D'),
         ('Surface', '齿面故障', '#d64b45', 'P')]
assert source['condition'] == '30_2'
assert source['window_size'] == 10485 and source['windows_per_file'] == 100
records = {(r['state'], r['window'], r['axis']): r for r in source['records']}
assert len(records) == len(source['records']) == 1500
for state, _, _, _ in CASES:
    for window in range(1, 101):
        aligned = [records[state, window, axis] for axis in 'xyz']
        assert len({(r['start'], r['stop']) for r in aligned}) == 1

plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],
                     'axes.unicode_minus':False,'font.size':11})

def points(ax, x_axis, y_axis, y_feature):
    for state, label, color, marker in CASES:
        values = np.array([[records[state, w, x_axis]['rms'], records[state, w, y_axis][y_feature]] for w in range(1,101)])
        assert values.shape == (100,2) and np.isfinite(values).all()
        ax.scatter(values[:,0], values[:,1], label=label, color=color,
                   marker=marker, s=38, alpha=.72, edgecolors='white', linewidths=.35)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax.margins(x=.08, y=.08)

for x_axis in ['x','y']:
    fig, ax = plt.subplots(figsize=(9,7))
    points(ax,x_axis,'z','rms')
    ax.set_xlabel(f'{x_axis} 方向 RMS（振动信号幅值）',labelpad=10)
    ax.set_ylabel('z 方向 RMS（振动信号幅值）',labelpad=10)
    fig.suptitle(f'30_2｜{x_axis} 与 z 方向 RMS 联合分布',x=.10,ha='left',y=.97,fontsize=22)
    fig.text(.10,.90,'每点对应同一窗口的两个方向 · 每种状态 100 点 · 每窗 10,485 点',fontsize=11,color='#52616c')
    handles, labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.53,.86),ncol=5,frameon=False,fontsize=10,handletextpad=.25,columnspacing=1)
    fig.text(.10,.025,'五种状态使用相同坐标尺度；展示全部窗口，未对特征归一化。',fontsize=10,color='#52616c')
    fig.tight_layout(rect=(.025,.065,.99,.80))
    fig.savefig(OUT / f'RMS-{x_axis}z散点图_30_2.png',dpi=180)
    plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(18,6.7),sharey=True)
for j,(ax,direction) in enumerate(zip(axes,'xyz')):
    points(ax,direction,direction,'crest_factor')
    ax.set_title(f'{direction} 方向 · 第 {j+2} 通道',loc='left',fontsize=16,pad=14)
    ax.set_xlabel('RMS（振动信号幅值）',labelpad=10)
    ax.tick_params(labelleft=True)
axes[0].set_ylabel('峰值因子（无量纲）',labelpad=10)
fig.suptitle('30_2｜RMS 与峰值因子的联合分布',x=.055,ha='left',y=.97,fontsize=25)
fig.text(.055,.895,'每个点代表一个窗口 · 每种状态 100 个窗口 · 每窗 10,485 点 · 不重叠',fontsize=12,color='#52616c')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.845),ncol=5,frameon=False,fontsize=12)
fig.text(.055,.035,'三幅图横轴分别取范围，纵轴范围一致；峰值因子 = 绝对峰值 / RMS。',fontsize=11,color='#52616c')
fig.tight_layout(rect=(.015,.09,.995,.78),w_pad=2.5)
fig.savefig(OUT/'RMS-峰值因子散点图_30_2.png',dpi=180)
plt.close(fig)
print('已生成 3 张图：x—z RMS、y—z RMS、三轴 RMS—峰值因子。每个面板 500 点；跨轴窗口边界一致。')
