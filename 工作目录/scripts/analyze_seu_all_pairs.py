"""30_2：24 项时域特征的全部 276 种双组合及描述性分布筛查。"""
from itertools import combinations
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '工作目录/outputs/SEU-数据直观认识'
OUT = BASE / '全部双特征组合'
CASES = [('Health', '健康', '#2878b5', 'o'),
         ('Chipped', '齿部故障', '#e69f00', 's'),
         ('Miss', '缺齿', '#239b73', '^'),
         ('Root', '齿根故障', '#9867b4', 'D'),
         ('Surface', '齿面故障', '#d64b45', 'P')]
FEATURES = [('mean', '均值'), ('std', '标准差'), ('rms', 'RMS'),
            ('peak', '绝对峰值'), ('peak_to_peak', '峰峰值'),
            ('skewness', '偏度'), ('kurtosis', '峭度'), ('crest_factor', '峰值因子')]
KEYS = [(a, f) for a in 'xyz' for f, _ in FEATURES]
NAMES = dict(FEATURES)


def read_matrix():
    source = json.loads((BASE / '窗口特征_30_2.json').read_text())
    assert source['condition'] == '30_2' and source['windows_per_file'] == 100
    assert source['window_size'] == 10485 and source['overlap'] == 0
    records = {(r['state'], r['window'], r['axis']): r for r in source['records']}
    assert len(records) == len(source['records']) == 1500
    rows = []
    for state, _, _, _ in CASES:
        for w in range(1, 101):
            axes = [records[state, w, a] for a in 'xyz']
            assert len({(r['start'], r['stop']) for r in axes}) == 1
            rows.append([records[state, w, a][f] for a, f in KEYS])
    values = np.array(rows)
    assert values.shape == (500, 24) and np.isfinite(values).all()
    labels = np.repeat(np.arange(5), 100)
    windows = np.tile(np.arange(1, 101), 5)
    return values, labels, windows


def short_name(i):
    a, f = KEYS[i]
    return f'{a} {NAMES[f]}'


def axis_name(i):
    unit = '无量纲' if KEYS[i][1] in ('skewness', 'kurtosis', 'crest_factor') else '振动信号幅值'
    return f'{short_name(i)}（{unit}）'


def neighbor_fraction(values, labels, mask):
    """同标签近邻占比；只描述已有点的混合程度，不训练分类器。"""
    v, y = values[mask], labels[mask]
    distances = np.sum((v[:, None, :] - v[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1, kind='stable')[:, :5]
    return float(np.mean(y[:, None] == y[neighbors]))


def screen(values, labels, windows, pairs):
    scale = values.std(axis=0)
    assert np.all(scale > 0)
    normalized = (values - values.mean(axis=0)) / scale
    masks = {
        'all_five': labels >= 0,
        'three_faults': labels >= 2,
        'three_faults_first_50': (labels >= 2) & (windows <= 50),
        'three_faults_last_50': (labels >= 2) & (windows > 50),
    }
    singles = [{name: neighbor_fraction(normalized[:, [i]], labels, mask)
                for name, mask in masks.items()} for i in range(24)]
    result = []
    for number, (i, j) in enumerate(pairs, 1):
        scores = {name: neighbor_fraction(normalized[:, [i, j]], labels, mask)
                  for name, mask in masks.items()}
        result.append(dict(
            pair_id=number, pdf_page=(number - 1) // 12 + 1,
            page_panel=(number - 1) % 12 + 1,
            features=[short_name(i), short_name(j)],
            feature_keys=[list(KEYS[i]), list(KEYS[j])],
            same_label_neighbor_fraction=scores,
            three_fault_gain_over_better_single=scores['three_faults'] - max(
                singles[i]['three_faults'], singles[j]['three_faults']),
            three_fault_min_half=min(scores['three_faults_first_50'], scores['three_faults_last_50']),
            pearson_correlation=float(np.corrcoef(values[:, i], values[:, j])[0, 1]),
        ))
    ranked = sorted(result, key=lambda r: (r['three_fault_min_half'],
                    r['same_label_neighbor_fraction']['three_faults']), reverse=True)
    report = dict(
        condition='30_2', rows=500, feature_count=24, pair_count=276,
        points_per_state=100, window_size=10485, overlap=0,
        scope='描述已观察窗口的分布；未训练模型，未进行训练/验证/测试评价。',
        method='每项特征使用全部500点的均值与总体标准差做一次标准化。每个子集中，每点找5个其他最近邻，计算同标签占比。前后半段沿用相同标准化尺度。散点图展示原始特征值。',
        focus_states=['Miss', 'Root', 'Surface'],
        ranking='按三类故障前后半段同类近邻占比的较小值降序；同分按三类全部窗口占比降序。排序用于安排看图顺序。',
        rank_pair_ids=[r['pair_id'] for r in ranked],
        single_features=[dict(feature=short_name(i), same_label_neighbor_fraction=s)
                         for i, s in enumerate(singles)],
        pairs=result,
    )
    (OUT / '全部组合_描述性筛查_30_2.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return result, ranked


def scatter(ax, values, labels, i, j, focus=False, compact=False):
    for code, (_, label, color, marker) in enumerate(CASES):
        if focus and code < 2:
            continue
        v = values[labels == code]
        ax.scatter(v[:, i], v[:, j], label=label, color=color, marker=marker,
                   s=15 if compact else 27, alpha=.72, edgecolors='white', linewidths=.25)
    ax.set_xlabel(axis_name(i), fontsize=9 if compact else 11, labelpad=6)
    ax.set_ylabel(axis_name(j), fontsize=9 if compact else 11, labelpad=6)
    ax.grid(alpha=.18)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8 if compact else 9)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(axis='both', style='sci', scilimits=(-3, 4), useMathText=True, useOffset=False)
    ax.margins(.08)


def legend(fig, y, size=11):
    handles = [Line2D([], [], linestyle='', marker=m, color=c, markersize=7, label=l)
               for _, l, c, m in CASES]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, y),
               ncol=5, frameon=False, fontsize=size)


def atlas(values, labels, pairs):
    path = OUT / '全部276种双特征组合_30_2.pdf'
    with PdfPages(path, metadata={'Title': 'SEU 30_2 | 全部276种双特征组合',
                                 'Subject': '500个已观察窗口的描述性联合分布'}) as pdf:
        for page in range(23):
            fig, axes = plt.subplots(4, 3, figsize=(16.5, 20))
            for panel, (i, j) in enumerate(pairs[page * 12: (page + 1) * 12]):
                ax = axes.flat[panel]
                number = page * 12 + panel + 1
                scatter(ax, values, labels, i, j, compact=True)
                ax.set_title(f'{number:03d}  {short_name(i)} × {short_name(j)}',
                             loc='left', fontsize=12, pad=12)
            fig.suptitle('30_2｜全部双特征组合', x=.06, ha='left', y=.975, fontsize=25)
            fig.text(.06, .95, '24 项特征 · 276 种配对 · 每图 500 点 · 每种状态 100 个窗口',
                     fontsize=12, color='#52616c')
            legend(fig, .935)
            fig.text(.06, .018, '每点来自同一窗口；图内五类共用坐标。各图坐标范围分别设置，展示原始特征值。',
                     fontsize=10, color='#52616c')
            fig.text(.94, .018, f'{page + 1} / 23', ha='right', fontsize=10, color='#52616c')
            fig.tight_layout(rect=(.025, .04, .985, .905), h_pad=3.4, w_pad=2.5)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def highlights(values, labels):
    idx = {key: i for i, key in enumerate(KEYS)}
    panels = [
        (('x', 'rms'), ('z', 'rms'), False, 'A  五种状态：x RMS + z RMS', '健康在右上方，齿部故障在左下方，其余三类位于中间。'),
        (('x', 'rms'), ('z', 'rms'), True, 'B  放大三类：x RMS + z RMS', '两个方向联合后，三类形成较清楚的分布带。'),
        (('y', 'rms'), ('z', 'rms'), True, 'C  三类：y RMS + z RMS', '齿根位置突出；缺齿与齿面故障局部交织。'),
        (('x', 'mean'), ('z', 'rms'), True, 'D  三类：x 均值 + z RMS', '新增观察对象；x 均值的前后段分布也有变化。'),
        (('x', 'kurtosis'), ('z', 'rms'), True, 'E  三类：x 峭度 + z RMS', '加入峭度后，缺齿与齿面故障仍有较多交织。'),
        (('x', 'std'), ('x', 'rms'), False, 'F  五种状态：x 标准差 + x RMS', '点几乎落在直线上，两项特征高度重复。'),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 17))
    for ax, (x, y, focus, title, caption) in zip(axes.flat, panels):
        scatter(ax, values, labels, idx[x], idx[y], focus=focus)
        ax.set_title(title, loc='left', fontsize=15, pad=32)
        ax.text(0, 1.04, caption, transform=ax.transAxes, fontsize=10, color='#52616c')
    fig.suptitle('30_2｜276 种双组合中的重点发现', x=.065, ha='left', y=.977, fontsize=25)
    fig.text(.065, .943, 'A、F 展示全部五类；B-E 只放大缺齿、齿根故障、齿面故障，每图 300 点。',
             fontsize=12, color='#52616c')
    legend(fig, .922, 12)
    fig.text(.065, .018, '每点对应一个窗口；各图坐标范围分别设置。结论针对当前 30_2 记录的已观察窗口。',
             fontsize=11, color='#52616c')
    fig.tight_layout(rect=(.025, .045, .985, .902), h_pad=3.5, w_pad=3.5)
    fig.savefig(OUT / '双特征组合_重点发现_30_2.png', dpi=180)
    plt.close(fig)


def mean_time_plot(values, labels, windows):
    i = KEYS.index(('x', 'mean'))
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)
    stats = []
    for ax, code in zip(axes, (2, 3, 4)):
        state, label, color, _ = CASES[code]
        v, w = values[labels == code, i], windows[labels == code]
        first, last = v[:50].mean(), v[50:].mean()
        ax.plot(w, v, color=color, lw=1.2)
        ax.hlines(first, 1, 50, color='#253342', lw=2)
        ax.hlines(last, 51, 100, color='#253342', lw=2, linestyles='--')
        ax.axvline(50.5, color='#8f9ba6', lw=.8, ls=':')
        ax.set_title(label, loc='left', fontsize=15)
        ax.set_xlabel('窗口序号（按采集顺序）')
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 4), useMathText=True)
        ax.set_xlim(1, 100)
        stats.append(dict(state=state, first_half_mean=float(first), last_half_mean=float(last),
                          first_half_std=float(v[:50].std()), last_half_std=float(v[50:].std())))
    axes[0].set_ylabel('x 均值（振动信号幅值）')
    fig.suptitle('30_2｜x 均值随窗口的变化', x=.065, ha='left', y=.98, fontsize=23)
    fig.text(.065, .88, '三图统一纵轴；黑色实线、虚线分别表示前 50 窗、后 50 窗的平均值。', fontsize=12, color='#52616c')
    fig.tight_layout(rect=(.02, .04, .985, .83), w_pad=2)
    fig.savefig(OUT / 'x均值_前后段观察_30_2.png', dpi=180)
    plt.close(fig)
    (OUT / 'x均值_前后段统计_30_2.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': ['PingFang SC', 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 11, 'pdf.fonttype': 3})
    values, labels, windows = read_matrix()
    pairs = list(combinations(range(24), 2))
    assert len(pairs) == len(set(pairs)) == 276
    result, ranked = screen(values, labels, windows, pairs)
    pdf = atlas(values, labels, pairs)
    highlights(values, labels)
    mean_time_plot(values, labels, windows)
    print(f'已完成 {len(result)} 种配对、23 页图册；每图 500 个点，全部共 138,000 个散点。')
    print(f'图册：{pdf}')
    print('三类故障前后半段描述性排序前八：')
    for r in ranked[:8]:
        print(r['features'], r['same_label_neighbor_fraction'])


if __name__ == '__main__':
    main()
