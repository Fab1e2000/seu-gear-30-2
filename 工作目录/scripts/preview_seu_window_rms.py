"""固定前四个时间块，对比候选窗长的 x-z 或 x-y RMS 分布。"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser()
parser.add_argument('--axes', choices=['xz','xy'], default='xz')
args=parser.parse_args()
second_axis=2 if args.axes=='xz' else 1
axis_label=args.axes[1]
R=Path(__file__).resolve().parents[1]
cfg=json.loads((R/'configs/seu_30_2_split.json').read_text())
raw=np.load(R/'outputs/SEU-RF训练/development_signals.npz')
O=R/'outputs/SEU-窗长比较';O.mkdir(exist_ok=True)
lengths=[1024,2048,4096,8192,10485]
cases=[('Health','健康','#2878b5'),('Chipped','齿部故障','#e69f00'),('Miss','缺齿','#239b73'),('Root','齿根故障','#9867b4'),('Surface','齿面故障','#d64b45')]
result={};counts=[]
for n in lengths:
 result[n]={}
 for state,_,_ in cases:
  signal=raw[state];assert signal.shape==(838848,3)
  parts=[]
  for key in cfg['final_train_blocks']:
   b=cfg['blocks'][key];count=(b['stop']-b['start'])//n
   x=signal[b['start']:b['start']+count*n][:,[0,second_axis]].reshape(count,n,2)
   parts.append(np.sqrt((x*x).mean(axis=1)))
  result[n][state]=np.concatenate(parts)
 counts.append({'window_size':n,'windows_per_block_per_state':209712//n,'windows_per_state':4*(209712//n),'total_windows':20*(209712//n),'tail_per_block':209712%n})
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
fig,axs=plt.subplots(2,5,figsize=(23,10))
for col,n in enumerate(lengths):
 for row in range(2):
  ax=axs[row,col]
  for state,label,color in cases:
   if row==1 and state not in ['Chipped','Surface']:continue
   x=result[n][state];ax.scatter(x[:,0],x[:,1],s=10 if n<4096 else 17,c=color,alpha=.45,label=label,edgecolors='none')
  ax.grid(alpha=.18);ax.set_xlabel('x RMS');ax.set_ylabel(f'{axis_label} RMS')
  if row==0:ax.set_title(f'{n:,} 点／窗\n每类 {counts[col]["windows_per_state"]} 窗',fontsize=15)
  if col==0:ax.legend(fontsize=10,loc='best')
for row in range(2):
 points=np.concatenate([v for n in lengths for k,v in result[n].items() if row==0 or k in ['Chipped','Surface']])
 low=points.min(0);high=points.max(0);pad=(high-low)*.06
 for ax in axs[row]:ax.set_xlim(low[0]-pad[0],high[0]+pad[0]);ax.set_ylim(low[1]-pad[1],high[1]+pad[1]);ax.ticklabel_format(style='sci',axis='both',scilimits=(-2,2))
fig.suptitle(f'30_2｜不同窗长下的 x—{axis_label} RMS 联合分布',fontsize=24,y=.985)
fig.text(.5,.934,'上排：五种状态　｜　下排：齿部与齿面放大对比　｜　同一排统一坐标范围',ha='center',fontsize=14)
fig.text(.5,.025,'固定使用前 80% 的四个时间块；各块独立切窗、无重叠，块末不足一窗的点舍弃；末尾 20% 不参与本图。',ha='center',fontsize=13)
fig.tight_layout(rect=(.01,.055,.99,.91),w_pad=2,h_pad=3)
fig.savefig(O/f'候选窗长_{args.axes}RMS联合分布.png',dpi=160);plt.close(fig)
np.savez_compressed(O/f'候选窗长_{args.axes}RMS.npz',**{f'{n}_{state}':v for n,d in result.items() for state,v in d.items()})
(O/'窗口数量.json').write_text(json.dumps(counts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(counts,ensure_ascii=False))
