"""计算 30_2 五种状态三轴完整记录的时域统计量并绘图。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '工作目录/outputs/SEU-数据直观认识'
OUT.mkdir(parents=True, exist_ok=True)
CASES = [('Health', '健康'), ('Chipped', '齿部故障'), ('Miss', '缺齿'), ('Root', '齿根故障'), ('Surface', '齿面故障')]
FEATURES = [
    ('mean', '均值', '平均偏移', '振动信号幅值'),
    ('std', '标准差', '围绕均值的波动大小', '振动信号幅值'),
    ('rms', '均方根 RMS', '整体振动强度', '振动信号幅值'),
    ('peak', '绝对峰值', '最大的瞬时绝对幅值', '振动信号幅值'),
    ('peak_to_peak', '峰峰值', '最大值与最小值之差', '振动信号幅值'),
    ('skewness', '偏度', '幅值分布的不对称性', '无量纲'),
    ('kurtosis', '峭度', '极端幅值的突出程度', '无量纲；正态参考值 3'),
    ('crest_factor', '峰值因子', '绝对峰值 / RMS', '无量纲'),
]
records = []
for name, label in CASES:
    source = ROOT / f'原始数据/SEU/gearbox/gearset/{name}_30_2.csv'
    with source.open() as f:
        for _ in range(16):
            next(f)
        x = np.loadtxt((line.replace('\\t', '\t') for line in f), usecols=(1, 2, 3))
    assert x.shape == (1048560, 3) and np.isfinite(x).all()
    mean = x.mean(axis=0)
    centered = x - mean
    std = np.sqrt(np.mean(centered ** 2, axis=0))
    rms = np.sqrt(np.mean(x ** 2, axis=0))
    peak = np.max(np.abs(x), axis=0)
    assert np.all(std > 0) and np.all(rms > 0)
    values = dict(mean=mean, std=std, rms=rms, peak=peak,
                  peak_to_peak=np.ptp(x, axis=0),
                  skewness=np.mean((centered / std) ** 3, axis=0),
                  kurtosis=np.mean((centered / std) ** 4, axis=0),
                  crest_factor=peak / rms)
    # 交叉核对总体二阶矩及先前独立计算的极值。
    assert np.allclose(rms ** 2, std ** 2 + mean ** 2)
    records.append(dict(state=name, label=label, source=source.name,
                        points_per_axis=len(x), channels=[2, 3, 4],
                        features={k:v.tolist() for k,v in values.items()}))
manifest = json.loads((OUT / '绘图清单_30_2.json').read_text())
for row, old in zip(records, manifest):
    assert row['source'] == old['file']
    assert np.allclose(row['features']['peak'], np.maximum(np.abs(old['minimum']), np.abs(old['maximum'])))
(OUT / '时域特征_30_2.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'], 'axes.unicode_minus':False, 'font.size':11})
fig, axes = plt.subplots(2, 4, figsize=(19, 10))
for ax, (key, title, meaning, unit) in zip(axes.flat, FEATURES):
    vals = np.array([r['features'][key] for r in records])
    if key in ('mean', 'skewness'):
        bound = np.abs(vals).max()
        ax.imshow(vals, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        normalized = np.abs(vals) / bound
    else:
        ax.imshow(vals, cmap='Blues', vmin=0, vmax=vals.max(), aspect='auto')
        normalized = vals / vals.max()
    for i in range(5):
        for j in range(3):
            value = f'{vals[i,j]:.2e}' if key == 'mean' else (f'{vals[i,j]:.4f}' if key in ('std','rms','peak','peak_to_peak') else f'{vals[i,j]:.3f}')
            ax.text(j, i, value, ha='center', va='center', color='white' if normalized[i,j] > .65 else '#142536', fontsize=11)
    ax.set_xticks(range(3), ['x / 通道 2','y / 通道 3','z / 通道 4'], fontsize=10)
    ax.set_yticks(range(5), [label for _,label in CASES])
    ax.set_title(f'{title}\n{meaning}', loc='left', fontsize=14, pad=12)
    ax.set_xlabel(unit, fontsize=10, color='#52616c')
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
fig.suptitle('30_2｜五种状态的三轴时域特征', x=.035, ha='left', fontsize=24, y=.98)
fig.text(.035,.918,'30 Hz–2 V 设置 · 每轴完整记录 1,048,560 点 · 未滤波、归一化或截段',fontsize=13,color='#52616c')
fig.text(.035,.025,'每格为一个状态、一个方向的整段统计值；色深仅在同一小图内比较。均值、偏度用红 / 蓝区分正负。',fontsize=12,color='#52616c')
fig.tight_layout(rect=(.01,.06,.995,.89), w_pad=2.2, h_pad=2.5)
fig.savefig(OUT / '时域特征对比_30_2.png', dpi=160)
plt.close(fig)
print(json.dumps([{r['state']: {k:r['features'][k] for k in ['rms','kurtosis','crest_factor']}} for r in records], ensure_ascii=False, indent=2))
