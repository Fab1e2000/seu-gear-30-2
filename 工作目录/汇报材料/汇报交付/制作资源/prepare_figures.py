from pathlib import Path
import numpy as np, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':['PingFang SC','Arial'],'font.size':17,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False})
states=['Health','Chipped','Miss','Root','Surface']; names=['健康','齿部','缺齿','齿根','齿面']; colors=['#3572A5','#DC8439','#349989','#9165AC','#C75060']
im=Image.open(OUT.parent/'.build/paper7.png'); w,h=im.size;im.crop((w*538/990,h*755/1320,w*898/990,h*958/1320)).save(OUT/'实验台.png')
a=np.load(ROOT/'outputs/SEU-RF训练/development_signals.npz'); rms=np.load(ROOT/'outputs/SEU-窗长比较/候选窗长_xzRMS.npz')
for st,n in zip(states,names):
 fig,axs=plt.subplots(3,1,figsize=(7,4.6),sharex=True)
 for j,ax in enumerate(axs):
  ax.plot(np.arange(2048)/5120,a[st][200000:202048,j],lw=.7,color=['#3572A5','#349989','#DC8439'][j]);ax.set_ylabel('xyz'[j]);ax.grid(alpha=.15)
 axs[-1].set_xlabel('时间 / s');fig.suptitle(n+'状态：三轴波形片段',fontsize=20);fig.tight_layout();fig.savefig(OUT/f'wave_{st}.png',dpi=200);plt.close(fig)
for window in [2048,8192]:
 fig,ax=plt.subplots(figsize=(7,4.8))
 for st,n,c in zip(states,names,colors):
  p=rms[f'{window}_{st}'];ax.scatter(p[:,0],p[:,1],s=15,c=c,label=n,alpha=.65,edgecolors='none')
 ax.set_xlabel('x 轴 RMS');ax.set_ylabel('z 轴 RMS');ax.set_title(f'{window} 点窗口',fontsize=22,pad=34);ax.legend(ncol=5,fontsize=11,loc='lower center',bbox_to_anchor=(.5,1.0));ax.set_xlim(.006,.012);ax.set_ylim(.009,.018);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(OUT/f'rms_{window}.png',dpi=200);plt.close(fig)
for tag,file in [('rf','SEU-RF训练/final8192/results/summary.json'),('cnn','SEU-CNN训练/final/results/summary.json')]:
 d=json.loads((ROOT/'outputs'/file).read_text());cm=np.array(d['confusion_matrix']);fig,ax=plt.subplots(figsize=(6.6,4.9));ax.imshow(cm/cm.sum(axis=1,keepdims=True),vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(5),names);ax.set_yticks(range(5),names);ax.set_xlabel('预测类别');ax.set_ylabel('真实类别');ax.spines[['top','right','left','bottom']].set_visible(False)
 for i in range(5):
  for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if i==j else '#34495E',fontsize=21)
 fig.tight_layout();fig.savefig(OUT/f'cm_{tag}.png',dpi=200);plt.close(fig)
print('Generated 10 figures')
