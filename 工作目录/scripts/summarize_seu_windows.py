"""30_2：100 个不重叠窗口的三轴时域特征与完整数值表。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '工作目录/outputs/SEU-数据直观认识'
CASES = [('Health','健康'),('Chipped','齿部故障'),('Miss','缺齿'),('Root','齿根故障'),('Surface','齿面故障')]
FEATURES = [('mean','均值'),('std','标准差'),('rms','RMS'),('peak','绝对峰值'),('peak_to_peak','峰峰值'),('skewness','偏度'),('kurtosis','峭度'),('crest_factor','峰值因子')]
SIZE, COUNT = 10485, 100
records, arrays = [], []
for name,label in CASES:
    with (ROOT / f'原始数据/SEU/gearbox/gearset/{name}_30_2.csv').open() as f:
        for _ in range(16): next(f)
        raw = np.loadtxt((s.replace('\\t','\t') for s in f),usecols=(1,2,3))
    assert raw.shape == (1048560,3) and np.isfinite(raw).all()
    x = raw[:SIZE*COUNT].reshape(COUNT,SIZE,3)
    mean=x.mean(axis=1)
    std=x.std(axis=1)
    rms=np.sqrt(np.mean(x*x,axis=1))
    z=(x-mean[:,None,:])/std[:,None,:]
    peak=np.abs(x).max(axis=1)
    vals=dict(mean=mean,std=std,rms=rms,peak=peak,peak_to_peak=np.ptp(x,axis=1),skewness=np.mean(z**3,axis=1),kurtosis=np.mean(z**4,axis=1),crest_factor=peak/rms)
    assert np.allclose(rms*rms,std*std+mean*mean)
    assert np.allclose(mean.mean(axis=0),raw[:SIZE*COUNT].mean(axis=0))
    assert np.allclose(peak.max(axis=0),np.abs(raw[:SIZE*COUNT]).max(axis=0))
    assert all(np.isfinite(v).all() for v in vals.values())
    arrays.append(vals)
    for i in range(COUNT):
        for j,axis in enumerate('xyz'):
            records.append(dict(state=name,label=label,window=i+1,axis=axis,channel=j+2,start=i*SIZE,stop=(i+1)*SIZE,**{k:float(v[i,j]) for k,v in vals.items()}))
meta=dict(condition='30_2',window_size=SIZE,windows_per_file=COUNT,overlap=0,sample_index='zero-based, stop exclusive',unused_tail_per_file=60,scope='exploratory; no train/validation/test split assigned',records=records)
(OUT/'窗口特征_30_2.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':10})
fig,axes=plt.subplots(8,3,figsize=(18,23),sharex=True,sharey='row')
colors=['#2878b5','#e69f00','#239b73','#9867b4','#d64b45']
for i,(key,label) in enumerate(FEATURES):
    for j,axis in enumerate('xyz'):
        ax=axes[i,j]
        for (_,state),vals,color in zip(CASES,arrays,colors):
            ax.plot(np.arange(1,101),vals[key][:,j],label=state,color=color,lw=1,alpha=.85)
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        if i==0: ax.set_title(f'{axis} 方向 · 第 {j+2} 通道',fontsize=16,pad=16)
        if j==0: ax.set_ylabel(label+'\n'+('振动信号幅值' if i<5 else '无量纲'),fontsize=12)
        if key=='mean': ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,3),useMathText=True)
        ax.set_xlim(1,100)
        ax.set_xticks([1,20,40,60,80,100])
        if i==7: ax.set_xlabel('窗口序号（按采集顺序）',fontsize=12)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.suptitle('30_2｜100 个窗口的三轴特征变化',fontsize=26,x=.065,ha='left',y=.987)
fig.text(.065,.962,'每窗 10,485 点 · 不重叠 · 每个文件末尾 60 点暂不使用 · 同一行统一纵轴范围',fontsize=13,color='#52616c')
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.951),ncol=5,frameon=False,fontsize=13)
fig.text(.065,.012,'每条曲线对应一种状态；窗口序号表示各文件内部的采集顺序。峭度采用正态参考值为 3 的定义。',fontsize=12,color='#52616c')
fig.tight_layout(rect=(.025,.03,.99,.925),h_pad=1.4,w_pad=1.7)
fig.savefig(OUT/'窗口特征变化_30_2.png',dpi=160)
plt.close(fig)
head=''.join(f'<th>{label}</th>' for _,label in FEATURES)
rows=[]
for r in records:
    cells=''.join(f'<td>{r[k]:.8g}</td>' for k,_ in FEATURES)
    rows.append(f"<tr><td>{r['label']}</td><td>{r['window']}</td><td>{r['axis']}</td><td>{r['start']}</td><td>{r['stop']}</td>{cells}</tr>")
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>30_2 窗口特征数值表</title><style>body{font:14px system-ui;margin:28px;color:#223}table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}th,td{padding:8px;border:1px solid #ddd;text-align:right}th{position:sticky;top:0;background:#eaf1f7}tr:nth-child(even){background:#f5f7fa}td:first-child{text-align:left}</style><h1>30_2｜窗口特征完整数值表</h1><p>五种状态 × 100 个窗口 × 三轴，共 1,500 行、12,000 个特征值。每窗 10,485 点，不重叠；末尾 60 点暂不使用。采样范围从 0 编号，包含起点、不包含终点。</p><p>均值、标准差、RMS、绝对峰值、峰峰值：振动信号幅值；偏度、峭度、峰值因子：无量纲。峭度正态参考值为 3。</p><table><thead><tr><th>状态</th><th>窗口</th><th>方向</th><th>起点</th><th>终点（不含）</th>'''+head+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></html>'
(OUT/'窗口特征数值表_30_2.html').write_text(html)
print(f'已计算 {len(records)} 行、{len(records)*len(FEATURES)} 个特征值；窗口覆盖 [0, {SIZE*COUNT})。')
