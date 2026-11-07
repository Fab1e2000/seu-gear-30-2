from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=R/'outputs/SEU-RF训练/final8192/results'
new=json.loads((O/'summary.json').read_text());old=json.loads((R/'outputs/SEU-RF训练/final/results/summary.json').read_text())
d=np.load(O/'test_predictions.npz');cm=np.zeros((5,5),int);np.add.at(cm,(d['y'],d['prediction']),1)
assert np.array_equal(cm,new['confusion_matrix']) and cm.sum(axis=1).tolist()==[25]*5
assert abs(np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1)))-new['macro_f1'])<1e-12
assert abs(np.trace(cm)/125-new['accuracy'])<1e-12
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
labels=['健康','齿部','缺齿','齿根','齿面']
fig,axs=plt.subplots(1,2,figsize=(13,5.5))
for ax,s,title in zip(axs,[old,new],['2048 点：510 个窗口','8192 点：125 个窗口']):
 c=np.array(s['confusion_matrix']);pct=c/c.sum(1,keepdims=True)*100
 ax.imshow(pct,cmap='Blues',vmin=0,vmax=100)
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{c[i,j]}\n{pct[i,j]:.1f}%',ha='center',va='center',fontsize=10,color='white' if pct[i,j]>55 else '#223344')
 ax.set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,xlabel='预测状态',ylabel='真实状态',title=f"{title}\n准确率 {s['accuracy']:.2%}｜宏 F1 {s['macro_f1']:.4f}")
fig.suptitle('30_2｜两种窗长各自调参后的留出评价',fontsize=17)
fig.text(.5,.015,'格内：窗口数与该真实类别的百分比；颜色统一为 0—100%。两轮超参数和评价窗口数量不同。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.05,1,.93));fig.savefig(O/'两种窗长混淆矩阵.png',dpi=170);plt.close(fig)
