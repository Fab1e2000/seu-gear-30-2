"""绘制五个连续时间块及三折时间交叉验证，不读取原始信号。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]


def main():
    cfg = json.loads((ROOT / '工作目录/configs/seu_30_2_split.json').read_text())
    blocks = cfg['blocks']
    counts = {}
    previous = 0
    for name, block in blocks.items():
        assert block['start'] == previous
        n, tail = divmod(block['stop'] - block['start'], cfg['window_size'])
        assert n == 102 and tail == 816
        counts[name] = n * len(cfg['source_files'])
        previous = block['stop']
    assert previous == cfg['samples_per_file']
    assert cfg['window_size'] == cfg['stride'] == 2048
    plt.rcParams.update({'font.family': ['PingFang SC', 'DejaVu Sans'], 'axes.unicode_minus': False})
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.subplots_adjust(left=.03, right=.98, top=.98, bottom=.03)
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis('off')
    ax.text(.2, 9.4, '30_2｜三折时间交叉验证与切窗', fontsize=24, weight='bold')
    ax.text(.2, 8.85, '五种状态各自按时间分块，再汇总对应窗口；最后 20% 留作测试', fontsize=14)
    colors = {'训练': '#3478c6', '验证': '#ba8318', '测试': '#8055b0', '暂不使用': '#e9edf2'}
    for j, (key, b) in enumerate(blocks.items()):
        x = 1.55 + j * 1.65
        ax.text(x + .77, 8.1, f'{key} · {j*20}—{(j+1)*20}%', ha='center', fontsize=13, weight='bold')
        ax.text(x + .77, 7.7, f"{b['start']+1:,}—{b['stop']:,}", ha='center', fontsize=10)
    rows=[]
    for i, fold in enumerate(cfg['folds'], 1):
        train, val = fold['train_blocks'], fold['validation_blocks']
        assert set(train).isdisjoint(val + cfg['test_blocks'])
        assert max(blocks[k]['stop'] for k in train) <= min(blocks[k]['start'] for k in val)
        assert max(blocks[k]['stop'] for k in val) <= blocks['B5']['start']
        rows.append((f'第 {i} 折', train, val))
        print(f"Fold {i}: train={sum(counts[k] for k in train)}, validation={sum(counts[k] for k in val)}")
    rows.append(('最终训练', cfg['final_train_blocks'], []))
    for i, (label, train, val) in enumerate(rows):
        y = 6.6-i*1.15
        ax.text(.2, y+.35, label, fontsize=14, va='center')
        for j, key in enumerate(blocks):
            status = '训练' if key in train else '验证' if key in val else '测试' if i == 3 and key in cfg['test_blocks'] else '暂不使用'
            x=1.55+j*1.65
            ax.add_patch(Rectangle((x,y),1.54,.72,facecolor=colors[status]))
            ax.text(x+.77,y+.36,status,ha='center',va='center',fontsize=14,color='#657080' if status=='暂不使用' else 'white')
    ax.text(.2, 2.35, '每个时间块：209,712 点 → 102 个完整窗口 × 2048 点 ＋ 816 点尾段不用',fontsize=14)
    ax.text(.2, 1.8, '每块五类共 510 窗；三折训练 510／1020／1530 窗，验证均为 510 窗。',fontsize=14)
    ax.text(.2, 1.25, '选好配置后：合并前四块的 2040 个窗口重新训练，再对测试块 510 窗评分。',fontsize=14)
    ax.text(.2, .65, '各块独立切窗，窗口不跨块；三轴与两个模型共用窗口。上方点号从 1 开始，两端包含。',fontsize=11,color='#657080')
    path=ROOT/'工作目录/outputs/SEU-分类实验/数据分段与切窗_30_2.png'
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160); plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
