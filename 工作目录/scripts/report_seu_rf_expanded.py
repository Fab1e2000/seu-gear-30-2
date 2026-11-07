"""核对扩展搜索结果，并生成六组对照图及简短笔记。"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; O=R/'outputs/SEU-RF训练/expanded-grid'; P=O/'results'
s=json.loads((P/'summary.json').read_text()); rank=json.loads((P/'ranking.json').read_text())
c=json.loads((R/'configs/seu_30_2_rf_expanded.json').read_text()); old=json.loads((R/'outputs/SEU-RF训练/results/summary.json').read_text())
rows=[json.loads(v) for v in (P/'fold_results.jsonl').read_text().splitlines()]
pred=np.load(P/'validation_predictions.npz'); data=np.load(P/'development_features.npz')
pred={k:pred[k] for k in pred.files}; data={k:data[k] for k in data.files}
assert len(rank)==len(c['configurations'])==1620 and len(rows)==s['fits']==4860 and not s['test_evaluated']
assert len({(r['id'],r['fold']) for r in rows})==4860
assert len({(v['feature_set'],json.dumps(v['parameters'],sort_keys=True)) for v in c['configurations']})==1620
for i,row in enumerate(rows):
 truth=data['y'][pred['validation_indices'][row['fold']-1]]; yp=pred['predictions'][i]
 cm=np.zeros((5,5),dtype=int);np.add.at(cm,(truth,yp),1)
 assert np.array_equal(cm,row['confusion_matrix']) and (cm.sum(axis=1)==102).all()
 score=np.mean(2*np.diag(cm)/(cm.sum(axis=0)+cm.sum(axis=1)))
 assert abs(score-row['macro_f1'])<1e-12
 assert abs(np.trace(cm)/cm.sum()-row['accuracy'])<1e-12
for r in rank:
 scores=[x['macro_f1'] for x in rows if x['id']==r['id']]
 assert np.allclose(scores,r['fold_macro_f1']) and abs(np.mean(scores)-r['mean_macro_f1'])<1e-12
 assert abs(np.std(scores)-r['std_macro_f1'])<1e-12
assert rank==sorted(rank,key=lambda r:(-r['mean_macro_f1'],r['id']))
plt.rcParams.update({'font.family':['PingFang SC','DejaVu Sans'],'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(15,6)); names=list('ABCDEF'); best=s['best_per_feature_set']
vals=[best[k]['mean_macro_f1'] for k in names]
axes[0].bar(names,vals,color='#3c80b5',alpha=.8,label='扩大搜索')
axes[0].scatter(list('ABC'),[old['best_per_feature_set'][k]['mean_macro_f1'] for k in 'ABC'],color='#d97d22',marker='D',zorder=4,label='上一轮')
for i,k in enumerate(names):
 axes[0].scatter([i]*3,best[k]['fold_macro_f1'],color='#193953',s=17,zorder=5)
 axes[0].text(i,vals[i]+.025,f'{vals[i]:.4f}',ha='center')
axes[0].set(ylim=(0,1.05),ylabel='宏平均 F1',title='各输入组调参后的最佳成绩（小圆点为各折）');axes[0].legend(loc='lower right')
conf={}
for k in names:
 cm=sum((np.array(r['confusion_matrix']) for r in rows if r['id']==best[k]['id']))
 conf[k]={'Miss_as_Surface':int(cm[2,4]),'Surface_as_Miss':int(cm[4,2])}
x=np.arange(6)
axes[1].bar(x-.18,[conf[k]['Miss_as_Surface'] for k in names],.36,label='缺齿 → 齿面')
axes[1].bar(x+.18,[conf[k]['Surface_as_Miss'] for k in names],.36,label='齿面 → 缺齿')
axes[1].set(xticks=x,xticklabels=names,ylabel='误判窗口数',title='最佳配置的三折验证汇总（每类共 306 窗）');axes[1].legend()
fig.suptitle('30_2｜六组输入 × 扩大超参数搜索',fontsize=20)
fig.text(.5,.02,'A：xz RMS   B：xyz RMS   C：24 特征   D：xyz 标准差   E：RMS＋峭度＋峰值因子   F：18 特征',ha='center',fontsize=11)
fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(O/'六组输入对比.png',dpi=160);plt.close(fig)
lines=['# 随机森林扩大搜索','', 'Windows 完成 1620 套配置、4860 次训练。沿用三折时间验证，测试段未使用。','', '![六组输入对比](../../outputs/SEU-RF训练/expanded-grid/六组输入对比.png)','', '| 输入 | 最佳宏 F1 | 三折标准差 | 相对上一轮 | 缺齿→齿面／齿面→缺齿 |','| --- | --- | --- | --- | --- |']
for k in names:
 b=best[k];delta=f"{b['mean_macro_f1']-old['best_per_feature_set'][k]['mean_macro_f1']:+.4f}" if k in 'ABC' else '新增'
 lines.append(f"| {k} | {b['mean_macro_f1']:.4f} | {b['std_macro_f1']:.4f} | {delta} | {conf[k]['Miss_as_Surface']}／{conf[k]['Surface_as_Miss']} |")
lines+=['','各组最佳参数：','', '| 输入 | 树数 | 深度 | 叶节点最少样本 | 分叉特征数 |','| --- | --- | --- | --- | --- |']
for k in names:
 p=best[k]['parameters'];lines.append(f"| {k} | {p['n_estimators']} | {p['max_depth'] or '不限制'} | {p['min_samples_leaf']} | {p['max_features']} |")
win=s['top_configuration'];lines+=['',f"本轮最高为 {win['feature_set']}，宏 F1 均值 {win['mean_macro_f1']:.4f}。这些成绩来自用于选择配置的验证段，最终测试留待后续。",'', '完整排名见 [ranking.csv](../../outputs/SEU-RF训练/expanded-grid/results/ranking.csv)。配置见 [扩展清单](../../configs/seu_30_2_rf_expanded.json)。远端目录：`D:\\SEU-training-20261104\\expanded-grid`。', '', '已核对全部折次、每折类别数量、预测对应的混淆矩阵、F1、准确率及汇总排名。']
(R/'历史实验/2048点/SEU-随机森林扩大搜索.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(s,ensure_ascii=False,indent=2))
