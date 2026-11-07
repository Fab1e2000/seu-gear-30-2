"""对照 30_2 三轴记录开头与中段的波形和分段均值。"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'工作目录/outputs/SEU-数据直观认识'
CASES=[('Health','健康'),('Chipped','齿部故障'),('Miss','缺齿'),('Root','齿根故障'),('Surface','齿面故障')]
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
# 前 20 个探索窗口；中段从第 51 个窗口开始，取相同长度。
length=209700
middle=524250
block=10485
for name,label in CASES:
    with (ROOT/f'原始数据/SEU/gearbox/gearset/{name}_30_2.csv').open() as f:
        for _ in range(16): next(f)
        raw=np.loadtxt((s.replace('\\t','\t') for s in f),usecols=(1,2,3))
    fig,axes=plt.subplots(3,3,figsize=(22,11),sharex='col')
    for j,direction in enumerate('xyz'):
        segments=[raw[:10000,j],raw[:length,j],raw[middle:middle+length,j]]
        blocks=[500,block,block]
        means=[a.reshape(20,b).mean(axis=1) for a,b in zip(segments,blocks)]
        mean_min=min(a.min() for a in means)
        mean_max=max(a.max() for a in means)
        pad=max((mean_max-mean_min)*.15,1e-5)
        bound=max(np.abs(a).max() for a in segments)*1.05
        for col,(start,signal,avg) in enumerate(zip([0,0,middle],segments,means)):
            ax=axes[j,col]
            # 全部原始点入图，分段均值用独立右轴显示细小偏移。
            ax.plot(np.arange(start,start+len(signal)),signal,color='#94b7d5',lw=.25,rasterized=True)
            ax.set_ylim(-bound,bound)
            ax.set_xlim(start,start+len(signal)-1)
            ax.set_ylabel(f'{direction} 方向 · 振动信号幅值')
            ax.grid(alpha=.15)
            ax.ticklabel_format(axis='x',style='sci',scilimits=(5,5),useMathText=True)
            twin=ax.twinx()
            twin.plot(start+np.arange(20)*blocks[col]+(blocks[col]-1)/2,avg,color='#bf3c2e',lw=2,marker='o',ms=3)
            twin.set_ylim(mean_min-pad,mean_max+pad)
            twin.set_ylabel('分段均值（红线，右轴）',color='#bf3c2e')
            twin.tick_params(axis='y',colors='#bf3c2e')
            twin.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
            ax.set_title(['开头细节：前 10,000 点','开头趋势：窗口 1—20','中段对照：窗口 51—70'][col]+f' · {direction} 方向',loc='left')
            if j==2: ax.set_xlabel('原始采样点序号（从 0 开始）')
    fig.suptitle(f'{label}（{name}）｜开头与中段对照',x=.07,ha='left',fontsize=23,y=.98)
    fig.text(.07,.928,'30_2 · 蓝色为原始波形；红色为分段均值（右轴）· 左列每 500 点求均值，中 / 右列每 10,485 点求均值',fontsize=12)
    fig.text(.07,.02,'同一方向三列采用相同的波形尺度和均值尺度。双轴用于放大均值变化，请分别读取刻度。',fontsize=11,color='#52616c')
    fig.tight_layout(rect=(.015,.045,.99,.9),h_pad=2,w_pad=2)
    fig.savefig(OUT/f'{name}_30_2_start_vs_middle.png',dpi=160)
    plt.close(fig)
    print(name, 'first/middle x,y,z means:',raw[:block].mean(axis=0),raw[middle:middle+block].mean(axis=0),flush=True)
