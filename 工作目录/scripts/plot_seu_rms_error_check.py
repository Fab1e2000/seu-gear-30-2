from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[1];o=r/'outputs/SEU-RF训练/final/results'
a=np.load(r/'outputs/SEU-RF训练/expanded-grid/results/development_features.npz');b=np.load(o/'test_predictions.npz');old=json.loads((r/'outputs/SEU-数据直观认识/窗口特征_30_2.json').read_text())
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(1,2,figsize=(13,5.5),sharex=True,sharey=True)
for state,label,color,idx in [('Chipped','齿部故障','#e69f00',1),('Surface','齿面故障','#d64b45',4)]:
 rows=[v for v in old['records'] if v['state']==state];xy=[[next(v['rms'] for v in rows if v['window']==w and v['axis']==axis) for axis in ['x','z']] for w in range(1,101)];xy=np.array(xy)
 axs[0].scatter(*xy.T,s=18,c=color,label=label,alpha=.7)
 x=a['X'][a['y']==idx][:,[2,18]];axs[1].scatter(*x.T,s=12,c=color,label=label+'（训练）',alpha=.3)
m=(b['y']==4)&(b['prediction']==1);axs[1].scatter(*b['X'][m][:,[2,18]].T,s=65,facecolors='none',edgecolors='#192e63',linewidths=1.5,label='齿面误判成齿部（15 个测试窗）')
axs[0].set_title('原探索图：10485 点／窗，全段数据');axs[1].set_title('正式模型：2048 点／窗，训练分布＋测试误判')
for ax in axs:
 ax.set_xlabel('x RMS');ax.legend(fontsize=9);ax.grid(alpha=.15)
axs[0].set_ylabel('z RMS');fig.tight_layout();fig.savefig(o/'RMS误判对照.png',dpi=160);plt.close(fig)
