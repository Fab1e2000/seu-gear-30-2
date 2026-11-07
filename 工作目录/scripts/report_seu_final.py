from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/'outputs/SEU-RF训练/final/results'
s=json.loads((o/'summary.json').read_text());d=np.load(o/'test_predictions.npz');cm=np.zeros((5,5),int)
np.add.at(cm,(d['y'],d['prediction']),1)
assert np.array_equal(cm,s['confusion_matrix']) and np.all(cm.sum(axis=1)==102)
assert abs(np.trace(cm)/510-s['accuracy'])<1e-12
assert abs(np.mean(2*np.diag(cm)/(cm.sum(axis=0)+cm.sum(axis=1)))-s['macro_f1'])<1e-12
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
labels=['健康','齿部故障','缺齿','齿根故障','齿面故障']
fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(cm,cmap='Blues',vmin=0,vmax=102)
ax.set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,xlabel='预测状态',ylabel='真实状态',title=f"最终测试｜准确率 {s['accuracy']:.2%} · 宏 F1 {s['macro_f1']:.4f}")
for i in range(5):
 for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>51 else '#223',fontsize=15)
fig.colorbar(im,ax=ax,label='窗口数');fig.tight_layout();fig.savefig(o/'测试混淆矩阵.png',dpi=160);plt.close(fig)
lines=['# 随机森林正式训练与测试','', '采用三轴 24 个特征，1000 棵树、最大深度 20、叶节点最少 1 个样本、分叉候选 4 个特征，种子 42。','', '用前 80% 的 2040 个窗口从头训练并保存模型，再用最后 20% 的 510 个窗口测试。每类测试窗口 102 个，测试后没有调参。','',f"测试准确率 **{s['accuracy']:.2%}**，宏平均 F1 **{s['macro_f1']:.4f}**，正确 {np.trace(cm)}／510 个窗口。",'', '![测试混淆矩阵](../../outputs/SEU-RF训练/final/results/测试混淆矩阵.png)','', '| 状态 | 精确率 | 召回率 | F1 |','| --- | --- | --- | --- |']
for name,key in zip(labels,['Health','Chipped','Miss','Root','Surface']):
 v=s['classification_report'][key];lines.append(f"| {name} | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1-score']:.4f} |")
lines+=['', '结果对应 30_2 同一工况、同次记录的后续片段。前面的全段探索已记录，当前测试成绩作为固定方案的最终评价。','', '模型与特征顺序：[random_forest.joblib](../../outputs/SEU-RF训练/final/results/random_forest.joblib)；完整指标：[summary.json](../../outputs/SEU-RF训练/final/results/summary.json)。','', '在 Windows 训练，远端目录 `D:\\SEU-training-20261104\\final`。模型重新加载后的预测一致，测试预测、混淆矩阵、准确率及宏 F1 已核对。']
(r/'历史实验/2048点/SEU-随机森林正式训练.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(s,ensure_ascii=False,indent=2))
