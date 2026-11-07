from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'outputs/SEU-转速调查';O.mkdir(exist_ok=True)
d=np.load(R/'outputs/SEU-RF训练/development_signals.npz');fs=5120.;n=65536;step=n//2;w=np.hanning(n);freq=np.fft.rfftfreq(n,1/fs)
cases=[('Health','健康'),('Chipped','齿部故障'),('Miss','缺齿'),('Root','齿根故障'),('Surface','齿面故障')]
psds={};records=[]
for state,label in cases:
 a=d[state];segs=np.stack([a[i:i+n] for i in range(0,len(a)-n+1,step)])
 segs-=segs.mean(axis=1,keepdims=True);p=(np.abs(np.fft.rfft(segs*w[None,:,None],axis=1))**2).mean(0)/(fs*(w*w).sum());p[1:-1]*=2;psds[state]=p
 for axis in range(3):
  peaks=np.flatnonzero((p[1:-1,axis]>p[:-2,axis])&(p[1:-1,axis]>p[2:,axis]))+1
  peaks=peaks[(freq[peaks]>=2)&(freq[peaks]<=100)]
  peaks=sorted(peaks,key=lambda k:p[k,axis],reverse=True)[:12]
  records.append({'state':state,'axis':'xyz'[axis],'peaks':[{'hz':float(freq[k]),'psd':float(p[k,axis])} for k in peaks]})
print(json.dumps(records,ensure_ascii=False))
(O/'频谱峰值.json').write_text(json.dumps({'assumed_fs':fs,'segment_points':n,'frequency_spacing':fs/n,'range_hz':[2,100],'records':records},ensure_ascii=False,indent=2))
np.savez_compressed(O/'功率谱.npz',frequency=freq,**psds)
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
fig,axs=plt.subplots(5,3,figsize=(17,15),sharex=True)
for i,(state,label) in enumerate(cases):
 for j in range(3):
  ax=axs[i,j];p=psds[state][:,j];sel=(freq>=2)&(freq<=100)
  ax.plot(freq[sel],10*np.log10(p[sel]/p[sel].max()),lw=.8,color='#3478b5')
  for f in [20,30,40,60,90]:ax.axvline(f,color='#d89530' if f%30 else '#c33',ls='--',alpha=.45,lw=.8)
  ax.set_ylim(-65,3);ax.grid(alpha=.15)
  if i==0:ax.set_title('xyz'[j]+' 方向')
  if j==0:ax.set_ylabel(label+'\n相对峰值 / dB')
  if i==4:ax.set_xlabel('频率 / Hz（按 5120 Hz 换算）')
fig.suptitle('30_2｜低频功率谱与转速候选',fontsize=22)
fig.text(.5,.945,'前 80% 信号 · Welch 平均 · 每段 65536 点 · 频率间隔 0.078125 Hz · 红虚线 30/60/90 Hz，橙虚线 20/40 Hz',ha='center',fontsize=11)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(O/'低频功率谱.png',dpi=150);plt.close(fig)
