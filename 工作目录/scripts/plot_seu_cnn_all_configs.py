"""48 组 CNN 参数组合的三折成绩；突出多种子复跑选中的配置。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT=Path(__file__).resolve().parents[1]/'outputs/SEU-CNN训练'
rows=json.loads((OUT/'ranking.json').read_text())
refined=json.loads((OUT/'refined_ranking.json').read_text())
winner=refined[0]; selected=winner['configuration']['id']
assert len(rows)==48 and len({r['configuration']['id'] for r in rows})==48
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False,'font.size':11})
fig,axes=plt.subplots(2,2,figsize=(21,14))
fig.subplots_adjust(left=.225,right=.955,top=.825,bottom=.21,wspace=1.25,hspace=.35)
foldcolors=['#67a9cf','#8766bb','#319a88']
minimum=min(f['macro_f1'] for r in rows for f in r['folds'])
low=np.floor((minimum-.02)*20)/20
for ax,w in zip(axes.flat,[1024,2048,4096,8192]):
    group=sorted([r for r in rows if r['configuration']['window']==w],key=lambda r:r['configuration']['id'])
    labels=[]
    for i,r in enumerate(group):
        c=r['configuration']; scores=[f['macro_f1'] for f in r['folds']]
        chosen=c['id']==selected
        if chosen: ax.axhspan(i-.45,i+.45,color='#fff0dc',zorder=0)
        ax.plot([min(scores),max(scores)],[i,i],color='#b7c3ce',lw=2,zorder=1)
        for j,score in enumerate(scores):
            ax.scatter(score,i+(j-1)*.16,s=19,color=foldcolors[j],zorder=3)
        ax.scatter(r['mean_macro_f1'],i,s=95 if chosen else 40,marker='*' if chosen else 'D',color='#da6a12' if chosen else '#253d54',zorder=4)
        ax.text(1.02,i,f"{r['mean_macro_f1']:.4f}",transform=ax.get_yaxis_transform(),va='center',fontsize=10,color='#b3500c' if chosen else '#253d54',weight='bold' if chosen else 'normal')
        labels.append(f"{c['id']}  |  {c['kernel']:2}  |  {'/'.join(map(str,c['channels']))}  |  {c['learning_rate']}")
    ax.set_yticks(range(12),labels)
    for tick,r in zip(ax.get_yticklabels(),group):
        if r['configuration']['id']==selected: tick.set_color('#b3500c');tick.set_weight('bold')
    ax.set_ylim(11.65,-.8); ax.set_xlim(low,1.008)
    ax.set_title(f'{w} 点窗口 · 12 组配置',loc='left',fontsize=16,pad=29)
    ax.text(-.04,1.025,'编号  |  卷积核  |  三层通道  |  学习率',ha='right',transform=ax.transAxes,color='#64748b',fontsize=10)
    ax.text(1.02,1.025,'均值',transform=ax.transAxes,color='#64748b',fontsize=10)
    ax.set_xlabel('验证宏 F1'); ax.grid(axis='x',color='#e8edf2');ax.set_axisbelow(True)
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0,pad=12)
fig.text(.055,.965,'CNN 全部参数组合对比',fontsize=27,weight='bold',color='#23354a')
fig.text(.055,.932,'48 组配置 × 三折时间验证｜主图统一使用种子 42，按窗长分组、按配置编号排列',fontsize=14,color='#526477')
handles=[Line2D([0],[0],marker='o',linestyle='',color=c,label=f'第 {i+1} 折') for i,c in enumerate(foldcolors)]
handles += [Line2D([0],[0],marker='D',linestyle='',color='#253d54',label='三折均值'),Line2D([0],[0],marker='*',markersize=13,linestyle='',color='#da6a12',label=f'最终选用 {selected}')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.05,.918),ncol=5,frameon=False)
fig.text(.055,.13,f"最终选用 {selected}：{winner['configuration']['window']} 点 · 卷积核 {winner['configuration']['kernel']} · 通道 {'→'.join(map(str,winner['configuration']['channels']))} · 学习率 {winner['configuration']['learning_rate']}",fontsize=18,weight='bold',color='#b3500c')
fig.text(.055,.098,'六组初筛并列第一的配置增加种子 7、2026 复跑后，按九次验证平均宏 F1 排序：',fontsize=13,color='#23354a')
parts=[f"{r['configuration']['id']}：{r['mean_macro_f1']:.5f}（标准差 {r['std_macro_f1']:.5f}）" for r in refined]
fig.text(.055,.072,'     '.join(parts[:3]),fontsize=12,color='#23354a')
fig.text(.055,.050,'     '.join(parts[3:]),fontsize=12,color='#23354a')
fig.text(.055,.024,'短横线连接三折最低与最高成绩；主图统一横轴。初筛并列第一的六组全部复跑；本图展示模型选择阶段的验证结果。',fontsize=11,color='#64748b')
fig.savefig(OUT/'CNN全部参数组合对比.png',dpi=180,facecolor='white')
fig.savefig(OUT/'CNN全部参数组合对比.pdf',facecolor='white')
plt.close(fig)
print(OUT/'CNN全部参数组合对比.png')
