"""核对正式评价，生成混淆矩阵与训练曲线，更新 CNN 过程记录。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/SEU-CNN训练/final'
s=json.loads((OUT/'results/summary.json').read_text());p=np.load(OUT/'results/test_predictions.npz')
cm=np.zeros((5,5),dtype=int);np.add.at(cm,(p['truth'],p['prediction']),1)
assert cm.tolist()==s['confusion_matrix'] and cm.sum(1).tolist()==[102]*5
assert np.isclose(np.trace(cm)/cm.sum(),s['accuracy'])
assert np.isclose(np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1))),s['macro_f1'])
h=json.loads((OUT/'results/history.json').read_text())
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(13,5.5),layout='constrained')
axes[0].plot([r['epoch'] for r in h],[r['train_loss'] for r in h],color='#287d9c')
axes[0].set(xlabel='训练轮数',ylabel='训练交叉熵损失',title=f"前 80% · 2040 窗 · {s['epochs']} 轮")
axes[0].grid(alpha=.15)
axes[1].imshow(cm/102*100,vmin=0,vmax=100,cmap='Blues')
labels=['健康','齿部','缺齿','齿根','齿面']
axes[1].set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,xlabel='预测状态',ylabel='真实状态',title=f"最后 20% · 510 窗 · 准确率 {s['accuracy']:.2%}")
for i in range(5):
 for j in range(5):axes[1].text(j,i,f'{cm[i,j]}\n{cm[i,j]/102:.1%}',ha='center',va='center',color='white' if cm[i,j]>51 else '#23354a')
fig.suptitle(f"CNN 正式训练与评价｜宏 F1 {s['macro_f1']:.4f}",fontsize=17)
fig.savefig(OUT/'CNN正式评价.png',dpi=180);plt.close(fig)
note=ROOT/'当前笔记/SEU-CNN全过程.md'
text=note.read_text().split('\n## 8. 正式训练与测试')[0]
text=text.replace('本轮完成模型选择；最终模型训练和保留段评价接着讨论。','调参完成后采用 CNN023，正式训练与评价见第 8 节。')
text+=f'''\n## 8. 正式训练与测试

采用 CNN023：2048 点窗口、卷积核 17、通道 16→32→64、学习率 0.001。Adam、批大小 32、交叉熵损失，其余设置沿用调参阶段。

九次交叉验证的最佳轮数为 {s['cv_best_epochs']}，取中位数 **{s['epochs']} 轮**。正式训练固定种子 42，使用前四个时间块的 **2040 窗**。三轴均值和标准差重新由前 80% 原始数据计算，窗口仍分别在各块内切分。

训练满 {s['epochs']} 轮后保存模型，再加载最后一个时间块，用同一套标准化参数评价 **510 窗，每类 102 窗**。测试成绩不参与选择轮数。模型重新加载后预测一致，混淆矩阵及指标已复算。

| 指标 | 结果 |
| --- | --- |
| 准确率 | {s['accuracy']:.2%} |
| 宏平均 F1 | {s['macro_f1']:.6f} |
| 正确／总窗口数 | {np.trace(cm)}／{cm.sum()} |

![CNN 正式训练与评价](../outputs/SEU-CNN训练/final/CNN正式评价.png)

'''
errors=[f'{labels[i]}→{labels[j]}：{cm[i,j]} 窗' for i in range(5) for j in range(5) if i!=j and cm[i,j]]
text+=('错分情况：'+'；'.join(errors)+'。' if errors else '五种状态的测试窗口均分类正确。')+'\n\n'
rf=json.loads((ROOT/'outputs/SEU-RF训练/final8192/results/summary.json').read_text())
text+=f'''随机森林与 CNN 使用相同的原始时间分块。随机森林采用 8192 点窗口，在 125 窗上的准确率为 {rf['accuracy']:.2%}、宏 F1 为 {rf['macro_f1']:.4f}；CNN 采用 2048 点窗口，在 510 窗上的准确率为 {s['accuracy']:.2%}、宏 F1 为 {s['macro_f1']:.4f}。两者窗口数量和尾段舍弃量不同，作为各自选定方案的效果比较。

最后 20% 此前已用于随机森林实验，记录为已查看的保留段。当前结论适用于同工况、同次连续记录内的时间分块评价。

[模型与评价结果](../outputs/SEU-CNN训练/final/results/) · [正式训练脚本](../scripts/train_seu_cnn_final.py)
'''
note.write_text(text)
print(json.dumps({k:s[k] for k in ['accuracy','macro_f1','epochs','training_seconds','confusion_matrix']},ensure_ascii=False))
