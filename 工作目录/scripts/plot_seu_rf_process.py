"""8192 点随机森林流程：整体、分块、联合调参。"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1];O=R/'outputs/SEU-流程说明';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
ink='#23354a';blue='#e9f2ff';gold='#fff2d8';grey='#f0f2f5'
def canvas(title,sub):
 f,a=plt.subplots(figsize=(14,8));f.subplots_adjust(0,0,1,1);a.set(xlim=(0,14),ylim=(0,8));a.axis('off');a.text(.6,7.4,title,fontsize=24,weight='bold',color=ink);a.text(.6,6.95,sub,fontsize=12,color='#5b697b');return f,a
def box(a,x,y,w,h,title,body,face=blue,dash=False):
 a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=0.12',facecolor=face,edgecolor='#8ca2ba',linestyle='--' if dash else '-',linewidth=1.2));a.text(x+w/2,y+h-.38,title,ha='center',va='center',fontsize=16,weight='bold',color=ink);a.text(x+w/2,y+.43,body,ha='center',va='center',fontsize=12,color=ink,linespacing=1.65)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':'#6b7d90','lw':1.8})
def save(f,name):f.savefig(O/name,dpi=160);plt.close(f)
f,a=canvas('① 随机森林实验：从观察到分类','当前方案：30_2 工况 · 五种状态 · 行星箱 x、y、z 三轴')
box(a,.6,4.6,3.7,1.8,'观察原始数据','三轴波形、八项时域特征\n多特征联合分布')
box(a,5.15,4.6,3.7,1.8,'比较窗长','同一时间范围比较候选\n选定 8192 点，不重叠')
box(a,9.7,4.6,3.7,1.8,'分块与切窗','五个时间块，各占 20%\n每块五类合计 125 窗')
arrow(a,4.35,5.5,5.05,5.5);arrow(a,8.9,5.5,9.6,5.5)
box(a,.6,1.8,3.7,1.8,'联合调参 · 已完成','3 组输入 × 36 套超参数\n每套做三折时间验证')
box(a,5.15,1.8,3.7,1.8,'候选范围内最佳','24 特征 · 500 棵树 · 深度 8\n验证宏 F1 均值 0.9893')
box(a,9.7,1.8,3.7,1.8,'正式训练与评价 · 已完成','前 80%：500 窗训练\n125 窗评价 · 准确率 99.20%',grey)
arrow(a,11.55,4.55,11.55,4.0);a.plot([11.55,.4,.4],[4,4,2.7],color='#6b7d90',lw=1.5);arrow(a,.4,2.7,.55,2.7)
arrow(a,4.35,2.7,5.05,2.7);arrow(a,8.9,2.7,9.6,2.7)
a.text(.6,.65,'末尾 20% 已在旧的 2048 点实验中查看；8192 点方案留出宏 F1 为 0.9920。',fontsize=12,color='#5b697b')
save(f,'01_整体流程.png')
f,a=canvas('② 时间分块与三折验证','每种状态各自按同一规则分块，再汇总五类窗口；每折重新训练一个森林')
for j in range(5):
 a.text(2.3+j*2.1,6.2,f'块 {j+1}｜{j*20}—{(j+1)*20}%',ha='center',fontsize=13,weight='bold')
 a.text(2.3+j*2.1,5.85,'每类 25 窗／五类 125 窗',ha='center',fontsize=10)
for i in range(4):
 y=4.7-i*1.0;a.text(.35,y+.3,f'第 {i+1} 折' if i<3 else '正式训练',fontsize=13)
 for j in range(5):
  label='训练' if j<=i else '验证' if j==i+1 and i<3 else '留出评价' if i==3 and j==4 else '本折不用'
  face=blue if label=='训练' else gold if label=='验证' else grey
  a.add_patch(FancyBboxPatch((1.35+j*2.1,y),1.9,.65,boxstyle='round,pad=0.02',facecolor=face,edgecolor='white'))
  a.text(2.3+j*2.1,y+.32,label,ha='center',va='center',fontsize=13)
a.text(.6,.95,'每块 209,712 点 = 25 × 8192 点 + 4912 点尾段（舍弃）。窗口不跨块。',fontsize=13,color=ink)
a.text(.6,.45,'三折训练／验证：125／125、250／125、375／125；正式训练 500 窗、评价 125 窗；末尾块在旧实验已查看。',fontsize=12,color='#5b697b')
save(f,'02_时间分块与三折.png')
f,a=canvas('③ 输入特征与超参数一起选择','每种输入都有同样的调参机会；验证成绩用于比较配置')
box(a,.6,4.7,4.2,1.8,'3 组输入','A：xz RMS（2 个）　B：xyz RMS（3 个）\nC：三轴八项特征（24 个）')
box(a,5.5,4.7,7.8,1.8,'每组 36 套超参数','树数：200／500　　深度：4／8／不限制\n叶节点最少样本：1／3／8　　分叉特征：sqrt／全部')
a.text(5.1,5.5,'×',fontsize=25,ha='center');arrow(a,7,4.65,7,4.1)
box(a,2.1,2.5,9.8,1.5,'108 套配置 × 3 折 = 324 次森林训练','每套得到三个宏 F1 → 求均值 → 比较排名',gold)
arrow(a,7,2.45,7,2.02)
a.text(7,1.65,'最佳三折均值：C  0.9893    >    B  0.9814    >    A  0.9578',ha='center',fontsize=18,weight='bold',color=ink)
a.text(7,1.1,'C 最佳：500 棵树 · 深度 8 · 叶节点最少 1 个样本 · 分叉候选 4 个特征 · 种子 42',ha='center',fontsize=12)
a.text(.6,.45,'每折仅 125 个验证窗口：24 特征略高，三轴 RMS 接近；正式训练采用 24 特征配置。',fontsize=12,color='#5b697b')
save(f,'03_联合调参.png')
print(O)
