"""核对并汇总 CNN 搜索结果。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'工作目录/outputs/SEU-CNN训练'

def main():
    ranking=json.loads((OUT/'ranking.json').read_text())
    refined=json.loads((OUT/'refined_ranking.json').read_text())
    assert len(ranking)==48
    records=list((OUT/'results').glob('*/metrics.json')); assert len(records)==180
    for path in records:
        r=json.loads(path.read_text()); pred=np.load(path.parent/'predictions.npz')
        cm=np.zeros((5,5),dtype=int)
        np.add.at(cm,(pred['truth'],pred['prediction']),1)
        assert cm.tolist()==r['confusion_matrix']
        assert cm.sum(1).tolist()==[209712//r['configuration']['window']]*5
        f1=np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1)))
        assert np.isclose(f1,r['macro_f1'])
    for row in ranking+refined:
        assert np.isclose(row['mean_macro_f1'],np.mean([f['macro_f1'] for f in row['folds']]))
    best=refined[0]; c=best['configuration']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for w in [1024,2048,4096,8192]:
        values=[r['mean_macro_f1'] for r in ranking if r['configuration']['window']==w]
        axes[0].scatter([str(w)]*len(values),values,alpha=.6)
    axes[0].set(xlabel='Window length',ylabel='3-fold macro F1',title='48 configurations / seed 42')
    for r in best['folds']:
        if r['seed']!=42: continue
        folder=OUT/'results'/f"{c['id']}_seed42_fold{r['fold']}"
        history=json.loads((folder/'history.json').read_text())
        axes[1].plot([h['epoch'] for h in history],[h['validation_loss'] for h in history],label=f"Fold {r['fold']}")
    axes[1].set(xlabel='Epoch',ylabel='Validation loss',title=f"Selected {c['id']} / seed 42")
    axes[1].legend(); fig.savefig(OUT/'CNN搜索结果.png',dpi=180); plt.close(fig)
    text='\n## 5. 调参结果\n\n完成 48 套配置的 144 次初筛训练，六组初筛并列第一的配置增加两个种子复跑，共 180 次。每折预测、混淆矩阵、宏 F1 及排名均已核对。\n\n'
    text+='| 配置 | 窗长 | 卷积核 | 通道 | 学习率 | 九次宏 F1 均值 | 标准差 |\n| --- | --- | --- | --- | --- | --- | --- |\n'
    for r in refined:
        q=r['configuration']; text+=f"| {q['id']} | {q['window']} | {q['kernel']} | {'→'.join(map(str,q['channels']))} | {q['learning_rate']} | {r['mean_macro_f1']:.4f} | {r['std_macro_f1']:.4f} |\n"
    text+='\n各窗长在种子 42 初筛中的最佳配置：\n\n| 窗长 | 配置 | 第一折 | 第二折 | 第三折 | 均值 |\n| --- | --- | --- | --- | --- | --- |\n'
    for w in [1024,2048,4096,8192]:
        r=next(r for r in ranking if r['configuration']['window']==w)
        text+=f"| {w} | {r['configuration']['id']} | "+' | '.join(f"{f['macro_f1']:.4f}" for f in r['folds'])+f" | {r['mean_macro_f1']:.4f} |\n"
    maximum=sum(json.loads(p.read_text())['epochs']==100 for p in records)
    text+=f'\n180 次训练中有 {maximum} 次达到 100 轮上限。本轮比较以这一训练预算为准。\n'
    text+='\n选中配置各折结果：\n\n| 随机种子 | 第一折宏 F1 | 第二折宏 F1 | 第三折宏 F1 |\n| --- | --- | --- | --- |\n'
    for seed in [42,7,2026]:
        scores=[r['macro_f1'] for r in best['folds'] if r['seed']==seed]
        text+=f"| {seed} | "+' | '.join(f'{s:.4f}' for s in scores)+' |\n'
    text+='\n![CNN 搜索结果](../outputs/SEU-CNN训练/CNN搜索结果.png)\n\n上述为前 80% 内的交叉验证成绩。本轮完成模型选择；最终模型训练和保留段评价接着讨论。\n'
    note=ROOT/'工作目录/当前笔记/SEU-CNN全过程.md'
    original=note.read_text().split('\n## 5. 调参结果')[0].replace('当前正在准备环境与试跑，完整成绩将在训练结束后汇入本文。','已完成环境配置、试跑和完整搜索，结果如下。')
    text+='\n## 6. 全部参数组合对比\n\n![CNN 全部参数组合对比](../outputs/SEU-CNN训练/CNN全部参数组合对比.png)\n'
    note.write_text(original.replace('完整搜索正在运行。', '完整搜索已完成。').replace('完整成绩将在训练结束后汇入本文。', '')+text)
    print(json.dumps(best,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
