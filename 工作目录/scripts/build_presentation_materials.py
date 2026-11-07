"""整理汇报图片，并生成可复用的 16:9 科学图与方法示意图。"""
from pathlib import Path
import json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R=Path(__file__).resolve().parents[1]; M=R/'汇报材料/PPT材料'; O=R/'outputs'
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False,'svg.fonttype':'path','font.size':12})
ink='#243b53'; blue='#e7f1fb'; green='#e4f3ed'; orange='#fff0da';grey='#edf0f4'
manifest=[]
def register(path,kind,source,usage):manifest.append(dict(file=str(path.relative_to(M)),type=kind,source=source,usage=usage))
def save(fig,folder,name,kind='方法示意',source='当前实验配置与训练脚本',usage='正文'):
    d=M/folder;d.mkdir(parents=True,exist_ok=True)
    for ext in ['png','svg']:
        p=d/f'{name}.{ext}';fig.savefig(p,dpi=170,facecolor='white');register(p,kind,source,usage)
    plt.close(fig)
def canvas(title,subtitle=''):
    f,a=plt.subplots(figsize=(16,9));f.subplots_adjust(0,0,1,1);a.set(xlim=(0,16),ylim=(0,9));a.axis('off')
    a.text(.65,8.35,title,fontsize=25,weight='bold',color=ink)
    a.text(.65,7.82,subtitle,fontsize=13,color='#587084');return f,a
def box(a,x,y,w,h,title,body='',color=blue,fs=15):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06,rounding_size=.12',facecolor=color,edgecolor='#b4c4d2'))
    a.text(x+w/2,y+h-.38,title,ha='center',va='center',fontsize=fs,weight='bold',color=ink)
    if body:a.text(x+w/2,y+h/2-.18,body,ha='center',va='center',fontsize=12,color=ink,linespacing=1.65)
def arrow(a,x,y,xx,yy):a.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#5c768d',lw=1.8))
def foot(a,s):a.text(.65,.45,s,fontsize=11,color='#587084')
# Existing figures: keep stages explicit.
for p in sorted(O.rglob('*.png')):
    rel=str(p.relative_to(O))
    if 'SEU-CNN训练' in rel: folder='06_CNN调参与评价'
    elif 'final8192' in rel or 'SEU-流程说明' in rel:folder='04_随机森林'
    elif 'SEU-RF训练' in rel:folder='08_附录_2048点历史实验'
    elif 'SEU-窗长比较' in rel or 'SEU-分类实验' in rel:folder='03_数据预处理与窗长'
    elif 'SEU-转速调查' in rel:folder='01_数据背景与频域'
    else:folder='02_数据直观认识'
    target=M/folder/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    register(target,'已有实验图',str(p.relative_to(R)),'附录' if folder.startswith('08') else '正文或附录')
# 1 task
f,a=canvas('研究任务：从三轴振动识别齿轮状态','SEU · 30_2 工况 · 五分类')
box(a,.8,3.5,3.5,2.3,'数据来源','DDS 数据集\n每种状态一个连续文件\n每文件 1,048,560 点')
box(a,6,3.5,3.5,2.3,'研究输入','行星箱 x、y、z 三轴\n对应原数据第 2—4 通道\n同步切片形成窗口')
box(a,11.2,2.9,3.8,3.5,'输出类别','健康 Health\n齿部 Chipped\n缺齿 Miss\n齿根 Root\n齿面 Surface',green)
arrow(a,4.4,4.6,5.85,4.6);arrow(a,9.6,4.6,11.05,4.6)
foot(a,'来源：原仓库说明与本地数据。研究暂以太阳轮故障为假设；原论文未明确故障安装部件。')
save(f,'01_数据背景与频域','01_任务与数据范围',source='来源核查/SEU-采样率与实验台核查.md；当前笔记/SEU-数据预处理.md')
# 2 routes
f,a=canvas('两条路线：人工特征与自动特征','共用原始时间分块，各自选择适合的窗长和模型配置')
box(a,.7,3.6,2.7,1.8,'三轴振动','同一批原始记录')
for y,title,body,end in [(5.4,'随机森林','8192 点 → 三轴 24 个特征','500 棵树 · 深度 8'),(2.2,'一维 CNN','2048 点 → 标准化三轴波形','三层卷积 → 全局平均池化')]:
    box(a,5,y,5.4,1.8,title,body,blue if y>3 else green);box(a,11.4,y,3.7,1.8,'五分类',end)
    arrow(a,3.5,4.5,4.85,y+.9);arrow(a,10.5,y+.9,11.25,y+.9)
foot(a,'两条路线均通过三折时间验证调参，再使用前 80% 正式训练、末尾 20% 评价。')
save(f,'07_两方案对比与结论','01_两路线整体框架')
# 3 partition
f,a=canvas('时间分块与三折向前验证','每类记录分成 5 块，每块 209,712 点；先分块，再在块内切窗')
for j in range(5):a.text(4+j*2.15,7.1,f'B{j+1}\n{j*20}—{(j+1)*20}%',ha='center',fontsize=13)
for i in range(4):
    a.text(.7,5.7-i*1.2,['第一折','第二折','第三折','正式训练／评价'][i],fontsize=14)
    for j in range(5):
        label='训练' if j<=i else '验证' if j==i+1 and i<3 else '保留评价' if i==3 and j==4 else '不用'
        box(a,3.1+j*2.15,5.35-i*1.2,1.8,.8,label,color=blue if label=='训练' else orange if label=='验证' else green if label=='保留评价' else grey,fs=13)
foot(a,'CNN（2048 点）：每折训练 510／1020／1530 窗，验证均为 510 窗；正式训练 2040 窗，保留评价 510 窗。')
save(f,'03_数据预处理与窗长','01_共用时间分块_CNN窗口数')
# 4 sample count
f,axes=plt.subplots(1,2,figsize=(16,9));f.subplots_adjust(top=.80,bottom=.18,wspace=.28)
w=[1024,2048,4096,8192];counts=[4080,2040,1020,500]
axes[0].bar(list(map(str,w)),counts,color=['#8aaac7','#d87932','#8aaac7','#8aaac7']);axes[0].set(xlabel='窗长（点）',ylabel='前 80% 的窗口数',title='各时间块独立切窗，无重叠')
for i,c in enumerate(counts):axes[0].text(i,c+60,str(c),ha='center')
axes[1].plot(w,np.array(w)/5120,'o-',color='#287d9c');axes[1].set(xticks=w,xlabel='窗长（点）',ylabel='单窗时长（秒）',title='观察时长与窗口数量的取舍');axes[1].grid(alpha=.2)
f.suptitle('窗长加长，单窗覆盖更多信号，可用窗口减少',fontsize=23,y=.93)
f.text(.08,.06,'按 5120 Hz 计算。短窗增加窗口数量，独立采集记录的数量保持不变。',fontsize=13)
save(f,'03_数据预处理与窗长','02_窗长_时长_样本量','配置计算','BLOCK=209712；每块 floor(BLOCK/L)；五类×四块')
# 5 normalization
f,a=canvas('逐轴标准化：训练侧计算，其他数据沿用','每折计算 3 个均值、3 个标准差；保留窗口之间的振幅差异')
box(a,.8,4.5,4.1,1.8,'该折训练块','五类采样点合在一起\nx、y、z 分别计算均值和标准差')
box(a,6.4,4.5,3.7,1.8,'统一变换','z = (x − μ训练) / σ训练',green)
box(a,11.6,4.5,3.5,1.8,'模型输入','形状：批大小 × 3 × L')
arrow(a,5,5.4,6.25,5.4);arrow(a,10.2,5.4,11.45,5.4)
box(a,1.5,1.9,5.5,1.6,'训练窗口','使用训练侧统计量');box(a,9,1.9,5.5,1.6,'验证／保留窗口','使用同一组统计量',orange)
arrow(a,8.2,4.4,4.2,3.6);arrow(a,8.2,4.4,11.7,3.6)
foot(a,'每个窗口不单独归一化。正式训练时，统计量重新由前 80% 的原始数据计算。')
save(f,'03_数据预处理与窗长','03_逐轴标准化')
# 6 features
f,a=canvas('随机森林的输入：三轴各八个时域特征','每个窗口最终对应 24 个数')
items=[('均值','信号平均位置'),('标准差','围绕均值的波动'),('RMS','整体幅值水平'),('绝对峰值','最大绝对幅值'),('峰峰值','最大值 − 最小值'),('偏度','分布的不对称性'),('峭度','分布尾部与极端值'),('峰值因子','绝对峰值 / RMS')]
for i,(title,body) in enumerate(items):box(a,.7+(i%4)*3.85,4.65-(i//4)*2.3,3.5,1.8,title,body)
foot(a,'对 x、y、z 分别计算；标准差采用总体定义，峭度采用 Pearson 定义。')
save(f,'04_随机森林','01_八个时域特征含义')
# 7 forest
f,a=canvas('随机森林：多棵树共同决定类别','机制示意；示意图中的树和票数不对应某次实际预测')
box(a,.7,3.6,3,1.8,'一个窗口','24 个时域特征')
for y,t in [(5.7,'树 1'),(3.5,'树 2'),(1.3,'其他树')]:
 box(a,5.2,y,3.8,1.6,t,'训练样本有放回抽样\n分叉时随机选择候选特征');arrow(a,3.8,4.5,5.05,y+.8);arrow(a,9.1,y+.8,10.8,4.5)
box(a,11,3.6,4,1.8,'汇总类别概率','取平均概率最大的类别',green)
foot(a,'本项目：500 棵树、最大深度 8、最小叶节点 1、每次分叉候选 4 个特征。')
save(f,'04_随机森林','02_随机森林分类原理','教学示意')
# 8 architecture
f,a=canvas('CNN023：三层一维卷积的完整结构','尺寸标记采用“通道数 × 序列长度”，省略批大小')
nodes=[('输入','3 × 2048'),('卷积 1 + ReLU','16 × 2048'),('最大池化','16 × 1024'),('卷积 2 + ReLU','32 × 1024'),('最大池化','32 × 512'),('卷积 3 + ReLU','64 × 512'),('全局平均池化','64 个特征'),('全连接','5 个类别分数')]
for i,(title,body) in enumerate(nodes):
 row=i//4; col=i%4 if row==0 else 3-i%4;x=.7+col*3.85;y=4.8 if row==0 else 2.0
 box(a,x,y,3.45,1.55,title,body,green if i==7 else blue,fs=14)
 if i<3:arrow(a,x+3.5,y+.75,x+3.8,y+.75)
 if i==3:arrow(a,x+1.7,y-.05,x+1.7,3.65)
 if i>3 and i<7:arrow(a,x-.08,y+.75,x-.35,y+.75)
foot(a,'卷积核均为 17，步长 1，两端各补 8；池化大小和步长均为 2。可学习参数共 44,773 个。')
save(f,'05_CNN结构与原理','01_CNN023网络结构')
# 9 convolution arithmetic
f,a=canvas('一个卷积核怎样产生一个响应值','教学算例：三通道、核长 3；实际模型第一层核长为 17')
box(a,.8,3.8,4.3,2.6,'同一位置的三轴片段','x：[1, 2, 3]\ny：[0, 1, 0]\nz：[2, 0, 1]')
box(a,5.85,3.8,4.3,2.6,'一个卷积核的权重','x：[1, 0, −1]\ny：[0, 2, 0]\nz：[1, 1, 0]')
box(a,11,3.8,4.1,2.6,'一个输出值','x：−2   y：2   z：2\n跨三轴相加，再加偏置 1\n响应值 = 3',green)
arrow(a,5.2,5.1,5.7,5.1);arrow(a,10.25,5.1,10.85,5.1)
a.text(1,2.2,'卷积核沿时间滑动 → 形成一条响应序列\n使用多个卷积核 → 形成多条不同的响应序列',fontsize=20,color=ink,linespacing=1.8)
foot(a,'每个输出通道都组合所有输入通道的信息；权重和偏置通过训练学习。算例数值仅用于说明计算。')
save(f,'05_CNN结构与原理','02_多通道卷积算例','教学示意')
# 10 activation/pool
f,a=canvas('ReLU 与最大池化：保留响应，压缩长度','教学算例；展示一条响应序列')
box(a,.8,4.2,4.3,2,'卷积输出','[−2, 3, 1, −1, 4, 2, −3, 5]')
box(a,5.85,4.2,4.3,2,'ReLU：负值变为 0','[0, 3, 1, 0, 4, 2, 0, 5]',green)
box(a,11,4.2,4.1,2,'最大池化：每两点取最大','[3, 1, 4, 5]',orange)
arrow(a,5.2,5.2,5.7,5.2);arrow(a,10.25,5.2,10.85,5.2)
a.text(1,2.3,'ReLU 引入非线性；最大池化将序列长度减半。\n池化大小和步长由训练前设定，池化本身没有可学习权重。',fontsize=19,color=ink,linespacing=1.8)
save(f,'05_CNN结构与原理','03_ReLU与池化算例','教学示意')
# 11 gap
f,a=canvas('从响应序列到五类分数','全局平均池化与最终全连接层做两种不同的计算')
box(a,.8,3.7,4,2.5,'64 条响应序列','每条长度为 512\n形状 64 × 512')
box(a,6,3.7,3.8,2.5,'逐条求平均','每条产生一个平均值\n得到 64 个特征',green)
box(a,11,3.7,4.2,2.5,'五个类别分别加权','每类使用全部 64 个特征\n64 × 5 权重 + 5 偏置\n得到 5 个类别分数')
arrow(a,4.9,4.95,5.85,4.95);arrow(a,9.9,4.95,10.85,4.95)
a.text(1,2.0,'池化：只沿每条序列的时间方向求平均。\n全连接：对 64 个特征加权求和；softmax 将五类分数转换为概率。',fontsize=18,color=ink,linespacing=1.8)
foot(a,'最后全连接层共有 325 个可学习参数。交叉熵训练直接接收类别分数。')
save(f,'05_CNN结构与原理','04_全局平均与五分类')
# 12 learning loop
f,a=canvas('CNN 的训练：用预测误差更新权重','训练批大小 32；Adam 优化器；交叉熵损失')
for x,t,b in [(.7,'输入与标签','一批三轴窗口\n对应真实类别'),(4.55,'前向计算','卷积 → 池化 → 分类\n输出五类分数'),(8.4,'计算损失','预测与真实标签比较\n交叉熵衡量误差'),(12.25,'反向传播','链式法则求梯度\nAdam 更新权重')]:box(a,x,4.4,3.1,2,t,b)
for x in [3.9,7.75,11.6]:arrow(a,x,5.4,x+.5,5.4)
a.plot([13.8,13.8,2.2,2.2],[4.3,3.4,3.4,4.2],color='#5c768d');arrow(a,2.2,3.8,2.2,4.3)
a.text(8,2.4,'遍历所有训练窗口一次 = 1 轮；每轮结束后在验证集计算损失',ha='center',fontsize=18,color=ink)
foot(a,'调参阶段：验证损失最低时保存模型，连续 15 轮无改善则停止，最多 100 轮。正式训练固定 62 轮。')
save(f,'05_CNN结构与原理','05_训练与反向传播流程')
# 13 chain simple
f,a=canvas('梯度下降的一个数值例子','教学算例：单个权重、平方误差损失；用于理解链式法则')
for x,t,b in [(.8,'输入','x = 2'),(4.5,'预测','ŷ = wx\nw = 1 → ŷ = 2'),(8.2,'目标','y = 4'),(11.9,'损失','L = (ŷ − y)² / 2\nL = 2')]:box(a,x,4.7,3.2,1.7,t,b)
a.text(1,3.5,'梯度：∂L/∂w = (ŷ − y) × x = (2 − 4) × 2 = −4',fontsize=20,color=ink)
a.text(1,2.5,'学习率 η = 0.1：新权重 w = 1 − 0.1 × (−4) = 1.4',fontsize=20,color=ink)
a.text(1,1.5,'重新预测：ŷ = 1.4 × 2 = 2.8，损失从 2 降到 0.72',fontsize=20,color='#218268')
foot(a,'实际 CNN 使用交叉熵与 Adam；此图采用简化的梯度下降计算说明更新方向。')
save(f,'05_CNN结构与原理','06_链式法则与梯度下降','教学示意')
# 14 search
f,a=canvas('CNN 联合调参与复跑','所有模型只在前 80% 的时间块内训练和验证')
box(a,.8,4.6,6.3,2.1,'48 组候选','4 种窗长 × 3 种卷积核\n× 2 种通道规模 × 2 种学习率')
box(a,9,4.6,6.1,2.1,'初筛 144 次训练','种子 42 × 48 组 × 三折\n按三折宏 F1 均值排序')
arrow(a,7.2,5.65,8.85,5.65)
box(a,.8,1.6,6.3,2.1,'六组初筛并列第一','增加种子 7、2026\n6 组 × 2 种子 × 三折 = 36 次')
box(a,9,1.6,6.1,2.1,'共 180 次训练','每个入选配置有九次验证成绩\n比较均值与波动，选择 CNN023',green)
arrow(a,12,4.5,4,3.8);arrow(a,7.2,2.65,8.85,2.65)
foot(a,'CNN023 与 CNN035 的均值几乎持平；保留较短的 2048 点配置，九次成绩波动也较小。')
save(f,'06_CNN调参与评价','01_调参流程_180次')
# 15 actual seed plot
refined=json.loads((O/'SEU-CNN训练/refined_ranking.json').read_text())
f,ax=plt.subplots(figsize=(16,9));f.subplots_adjust(top=.82,bottom=.2)
for i,r in enumerate(refined):
 vals=[q['macro_f1'] for q in r['folds']];ax.scatter(i+np.linspace(-.12,.12,9),vals,c='#d87932' if i==0 else '#7ba3c0',s=60,alpha=.8)
 ax.plot([i-.2,i+.2],[np.mean(vals)]*2,color=ink,lw=3)
ax.set(xticks=range(6),xticklabels=[r['configuration']['id'] for r in refined],ylabel='验证宏 F1',ylim=(.97,1.002));ax.grid(axis='y',alpha=.2)
f.suptitle('六组并列第一配置：九次验证成绩对比',fontsize=23,y=.94)
f.text(.08,.07,'每点对应一个“种子 × 时间折”；黑线为均值。CNN023 与 CNN035 几乎持平，橙色为最终采用配置。',fontsize=13)
save(f,'06_CNN调参与评价','02_六组复跑分布','实验结果','outputs/SEU-CNN训练/refined_ranking.json')
#16 validation best epoch
best=refined[0];record=next(q for q in best['folds'] if q['seed']==42 and q['fold']==2)
h=json.loads((O/f"SEU-CNN训练/results/CNN023_seed42_fold2/history.json").read_text())
f,ax=plt.subplots(figsize=(16,9));f.subplots_adjust(top=.82,bottom=.20)
ax.semilogy([q['epoch'] for q in h],[q['train_loss'] for q in h],label='训练损失',color='#287d9c')
ax.semilogy([q['epoch'] for q in h],[q['validation_loss'] for q in h],label='验证损失',color='#d87932')
ax.axvline(record['best_epoch'],color='#218268',ls='--',label=f"恢复第 {record['best_epoch']} 轮模型")
ax.axvline(record['epochs'],color='#64748b',ls=':',label=f"第 {record['epochs']} 轮停止")
ax.set(xlabel='训练轮数',ylabel='交叉熵损失（对数轴）');ax.legend();ax.grid(alpha=.15)
f.suptitle('早停的实际例子：CNN023 · 种子 42 · 第二折',fontsize=23,y=.94)
f.text(.08,.07,'验证损失第 66 轮最低；后续 15 轮未改善，第 81 轮停止，并恢复第 66 轮权重。',fontsize=13)
save(f,'06_CNN调参与评价','03_早停与最佳模型','实验结果','outputs/SEU-CNN训练/results/CNN023_seed42_fold2/history.json')
#17 epoch selection
values=[r['best_epoch'] for r in best['folds']]
f,ax=plt.subplots(figsize=(16,9));f.subplots_adjust(top=.82,bottom=.23)
ax.bar(range(9),values,color='#8aabc5');ax.axhline(62,color='#d87932',lw=2,label='中位数 62 轮')
ax.set(xticks=range(9),xticklabels=[f"种子 {r['seed']}\n第 {r['fold']} 折" for r in best['folds']],ylabel='最佳训练轮数',ylim=(0,115));ax.legend()
for i,v in enumerate(values):ax.text(i,v+2,str(v),ha='center')
f.suptitle('正式训练轮数由开发侧验证记录确定',fontsize=23,y=.94)
f.text(.08,.07,'九次最佳轮数的中位数为 62。正式训练使用全部前 80%，固定 62 轮，随后评价最后 20%。',fontsize=13)
save(f,'06_CNN调参与评价','04_正式训练轮数依据','实验结果','outputs/SEU-CNN训练/refined_ranking.json')
#18 final route
f,a=canvas('CNN 正式训练与测试','采用 CNN023，随机种子 42')
box(a,.8,4.2,4.1,2.2,'前 80%','2040 个窗口\n拟合三轴标准化参数\n固定训练 62 轮')
box(a,6,4.2,4.1,2.2,'保存并重载模型','固定网络权重\n保存均值、标准差与配置\n预测一致性核对')
box(a,11.2,4.2,4.1,2.2,'末尾 20%','510 个窗口，每类 102 个\n全部识别正确\n准确率 100% · 宏 F1 1.0000',green)
arrow(a,5,5.3,5.85,5.3);arrow(a,10.2,5.3,11.05,5.3)
a.text(1,2.3,'测试段只在模型保存后加载，测试成绩不参与选择训练轮数。',fontsize=20,color=ink)
foot(a,'末尾段已在随机森林实验中查看。当前评价范围：同工况、同次连续记录内的时间分块。')
save(f,'06_CNN调参与评价','05_正式训练与评价流程')
#19 comparison cm
rf=json.loads((O/'SEU-RF训练/final8192/results/summary.json').read_text());cnn=json.loads((O/'SEU-CNN训练/final/results/summary.json').read_text())
f,axes=plt.subplots(1,2,figsize=(16,9));f.subplots_adjust(top=.80,bottom=.20,wspace=.28)
labels=['健康','齿部','缺齿','齿根','齿面']
for ax,s,title in zip(axes,[rf,cnn],['随机森林 · 8192 点 · 125 窗','CNN · 2048 点 · 510 窗']):
 cm=np.array(s['confusion_matrix']);pct=cm/cm.sum(1)[:,None]*100
 ax.imshow(pct,vmin=0,vmax=100,cmap='Blues');ax.set(xticks=range(5),yticks=range(5),xticklabels=labels,yticklabels=labels,xlabel='预测类别',ylabel='真实类别',title=title)
 for i in range(5):
  for j in range(5):ax.text(j,i,f'{cm[i,j]}\n{pct[i,j]:.1f}%',ha='center',va='center',color='white' if pct[i,j]>50 else ink)
f.suptitle('两条路线在同一末尾时间段上的评价',fontsize=24,y=.94)
f.text(.07,.07,'按真实类别归一化。两条路线窗长、窗口数和尾段舍弃量不同，比较各自选定的完整方案。',fontsize=13)
save(f,'07_两方案对比与结论','02_RF与CNN混淆矩阵','实验结果','RF final8192 与 CNN final 的 summary.json')
#20 table summary
f,a=canvas('实验结果与适用范围','人工特征 + 随机森林，与三轴波形 + CNN')
columns=['项目','随机森林','CNN']; table=[['输入','三轴 24 个时域特征','标准化三轴原始波形'],['窗长','8192 点 / 1.6 秒','2048 点 / 0.4 秒'],['训练／评价窗口','500 / 125','2040 / 510'],['准确率',f"{rf['accuracy']:.2%}",f"{cnn['accuracy']:.2%}"],['宏 F1',f"{rf['macro_f1']:.4f}",f"{cnn['macro_f1']:.4f}"],['正确窗口','124 / 125','510 / 510']]
t=a.table(cellText=table,colLabels=columns,bbox=[.06,.28,.88,.52],cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(16)
for (row,col),cell in t.get_celld().items():cell.set_edgecolor('#d6e1eb');cell.set_facecolor(blue if row==0 else '#f8fafc' if row%2 else 'white')
a.text(1,1.45,'当前结果：两条路线均能识别这批数据中的五类状态。',fontsize=19,color=ink)
foot(a,'范围：30_2 同工况、同次记录的时间分块评价。进一步研究可增加独立运行记录或跨工况评价。')
save(f,'07_两方案对比与结论','03_结果汇总','实验结果','两条路线正式评价 summary.json')
#21 F1
f,a=canvas('五分类的宏平均 F1 怎样计算','教学算例；五类在最终平均中具有相同权重')
box(a,.8,4.2,4.1,2,'对每一类计算','精确率 P = TP / (TP + FP)\n召回率 R = TP / (TP + FN)')
box(a,6,4.2,4.1,2,'得到该类 F1','F1 = 2PR / (P + R)\n也可写为 2TP / (2TP + FP + FN)')
box(a,11.2,4.2,4.1,2,'五类取平均','宏 F1 = 五类 F1 之和 / 5',green)
arrow(a,5,5.2,5.85,5.2);arrow(a,10.2,5.2,11.05,5.2)
a.text(1,2.2,'例：五类 F1 为 1.00、0.90、0.80、0.95、0.85\n宏 F1 = 4.50 / 5 = 0.90',fontsize=21,color=ink,linespacing=1.8)
foot(a,'TP：该类判对；FP：其他类误判成该类；FN：该类误判成其他类。算例数值与实验结果分开。')
save(f,'09_答辩原理补充','01_宏平均F1原理','教学示意')
# Source evidence references, no unverified illustration used as actual apparatus.
d=M/'10_来源资料';d.mkdir(exist_ok=True)
for name in ['SEU-采样率与实验台核查.md','SEU-原始论文-Shao-2019.pdf','SEU-Issue3-用户提供的实验说明.jpg']:
 p=R/'来源核查'/name
 if p.exists():shutil.copy2(p,d/name);register(d/name,'来源资料',str(p.relative_to(R)),'核查；Issue 截图出处不完整，不作已确认设备参数图')
p=O/'SEU-数据直观认识/全部双特征组合/全部276种双特征组合_30_2.pdf'
if p.exists():
 d=M/'02_数据直观认识/双特征全组合附录';d.mkdir(exist_ok=True);shutil.copy2(p,d/p.name);register(d/p.name,'已有实验图',str(p.relative_to(R)),'附录；全组合检索')
# Render the relevant original-paper pages as source evidence.
import subprocess
subprocess.run(['pdftoppm','-f','7','-l','9','-r','120','-png',str(R/'来源核查/SEU-原始论文-Shao-2019.pdf'),str(M/'10_来源资料/原论文页')],check=True)
for page,desc in [('07','期刊页 2452：Fig. 8 实验台照片；同页其他图属于轴承数据'),('08','期刊页 2453：故障类型与工况说明'),('09','期刊页 2454：五分类与原论文实验设置')]:
    register(M/f'10_来源资料/原论文页-{page}.png','原论文页面',f'Shao 等，2019，PDF 第 {int(page)} 页',desc+'；文中分数属于原论文实验')
(M/'图片清单.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
lines=['# PPT 图片材料','', '按“数据来源 → 直观认识 → 数据处理 → 两条方法 → 调参 → 正式评价 → 对比结论”组织。新增图为 16:9，提供 PNG 与 SVG；已有大图保留原尺寸，适合放大或作为附录。','', '优先使用第 01、03、04、05、06、07 目录中的主线图。第 08 目录是历史 2048 点随机森林实验，第 09 是答辩原理补充，第 10 是来源核查。','', '实际数据图、配置计算和教学示意均在下表标明；教学示意中的数值不作实验结果。保留段已在前期实验查看，结果用于同工况、同次记录内比较。','', '| 文件 | 类型 | 依据 | 用途 |','| --- | --- | --- | --- |']
for r in manifest:
 if r['file'].endswith('.svg'):continue
 lines.append(f"| [{r['file']}]({r['file']}) | {r['type']} | {r['source']} | {r['usage']} |")
lines+=['','图片生成脚本：[build_presentation_materials.py](../../scripts/build_presentation_materials.py)。修改源数据或配置后可重新生成。','']
(M/'README.md').write_text('\n'.join(lines))
# Contact sheet for visual QA; thumbnails only, not a slide asset.
imgs=[r for r in manifest if r['file'].endswith('.png') and r['type'] not in ['已有实验图','原论文页面']]
f,axs=plt.subplots(int(np.ceil(len(imgs)/4)),4,figsize=(20,3*int(np.ceil(len(imgs)/4))))
for ax,r in zip(axs.flat,imgs):ax.imshow(plt.imread(M/r['file']));ax.set_title(Path(r['file']).stem,fontsize=9);ax.axis('off')
for ax in list(axs.flat)[len(imgs):]:ax.axis('off')
f.tight_layout();f.savefig(M/'新增图片总览.png',dpi=100);plt.close(f)
print(json.dumps({'materials':str(M),'new_figures':len(imgs),'indexed_files':len(manifest)},ensure_ascii=False))
