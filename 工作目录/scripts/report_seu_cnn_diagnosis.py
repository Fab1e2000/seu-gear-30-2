"""复算诊断指标、导出汇总表和报告图。"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/SEU-CNN过拟合诊断'
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
d=json.loads((OUT/'summary.json').read_text());rows=d['results']
assert len(rows)==96
for r in rows:
    p=np.load(OUT/f"{r['arm']}_seed{r['seed']}"/f"{r['policy']}_{r['block']}.npz")
    t,pred=p['truth'],p['prediction'];cm=np.zeros((5,5),dtype=int)
    np.add.at(cm,(t,pred),1)
    assert np.array_equal(cm,r['confusion_matrix']) and int((t!=pred).sum())==r['errors']
    f1=np.mean(2*cm.diagonal()/(cm.sum(0)+cm.sum(1)))
    assert abs(f1-r['macro_f1'])<1e-12 and abs((t==pred).mean()-r['accuracy'])<1e-12
    ce=-np.log(p['probability'][np.arange(len(t)),t]).mean()
    assert np.isclose(ce,r['loss'],atol=1e-6)
    if r['arm']=='baseline' and r['policy']=='fixed62':
        old=np.load(ROOT/'outputs/SEU-CNN时间间隔验证'/f"predictions_seed{r['seed']}_{r['block']}.npz")
        assert np.array_equal(old['prediction'],pred)
        assert np.allclose(old['probability'],p['probability'],atol=1e-6)
summary=[]
for arm in d['plan']['arms']:
    for policy in ['fixed62','early']:
        for b in ['B1','B2','B3','B4']:
            rr=[r for r in rows if r['arm']==arm['name'] and r['policy']==policy and r['block']==b]
            z=dict(arm=arm['name'],policy=policy,block=b,parameters=rr[0]['parameters'])
            for metric in ['accuracy','macro_f1','loss']:
                z[metric+'_mean']=float(np.mean([r[metric] for r in rr]))
                z[metric+'_sd']=float(np.std([r[metric] for r in rr]))
            z['errors']=' / '.join(str(r['errors']) for r in rr)
            z['epochs']=' / '.join(str(r['epoch']) for r in rr)
            summary.append(z)
with (OUT/'aggregate.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]);w.writeheader();w.writerows(summary)
print(json.dumps(summary,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for arm in d['plan']['arms']:
    for ax,policy in zip(axs,['fixed62','early']):
        rr=[r for r in summary if r['arm']==arm['name'] and r['policy']==policy and r['block']!='B1']
        ax.errorbar([0,1,2],[r['accuracy_mean']*100 for r in rr],
                    yerr=[r['accuracy_sd']*100 for r in rr],marker='o',capsize=3,label=arm['name'])
        ax.set(xticks=[0,1,2],xticklabels=['B2: 0.16 s','B3: 41.12 s','B4: 82.08 s'],
               ylabel='Accuracy (%)',title=policy+' (mean +/- seed SD)')
        ax.grid(alpha=.2);ax.legend()
fig.savefig(FIG/'accuracy.png',dpi=180);plt.close(fig)
fig,axs=plt.subplots(4,3,figsize=(12,11),layout='constrained')
for i,arm in enumerate(d['plan']['arms']):
    for j,seed in enumerate([42,7,2026]):
        h=json.loads((OUT/f"{arm['name']}_seed{seed}"/'history.json').read_text())
        ax=axs[i,j]
        for key,label in [('train_loss','B1 train'),('validation_loss','B2 validation')]:
            ax.plot([r['epoch'] for r in h],[r[key] for r in h],label=label)
        chosen=next(r['epoch'] for r in rows if r['arm']==arm['name'] and r['seed']==seed and r['policy']=='early')
        ax.axvline(chosen,color='green',ls=':',label='B2 best');ax.axvline(62,color='gray',ls='--',label='62')
        ax.set(yscale='log',title=f"{arm['name']}, seed {seed}",xlabel='Epoch',ylabel='Cross entropy');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=7);fig.savefig(FIG/'learning_curves.png',dpi=160);plt.close(fig)
with (OUT/'window_features.csv').open() as f:features=list(csv.DictReader(f))
shifts=[]
for state in ['Health','Chipped','Miss','Root','Surface']:
    for feature in ['x_mean','y_mean','z_mean','x_rms','y_rms','z_rms']:
        arrays=[np.array([float(r[feature]) for r in features if r['state']==state and r['block']==f'B{b}']) for b in range(1,5)]
        shifts.append(dict(state=state,feature=feature,**{f'B{i+1}_mean':float(a.mean()) for i,a in enumerate(arrays)},
                           B4_minus_B1_in_B1_window_sd=float((arrays[3].mean()-arrays[0].mean())/arrays[0].std())))
with (OUT/'feature_shifts.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=shifts[0]);w.writeheader();w.writerows(shifts)
fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,axis in zip(axs,'xyz'):
    for state in ['Health','Chipped','Miss','Root','Surface']:
        r=next(r for r in shifts if r['state']==state and r['feature']==axis+'_rms')
        ax.plot(range(1,5),[r[f'B{i}_mean']/r['B1_mean'] for i in range(1,5)],marker='o',label=state)
    ax.set(title=f'{axis.upper()} RMS relative to B1',xticks=range(1,5),xlabel='Block',ylabel='Mean window RMS / B1')
    ax.grid(alpha=.2)
axs[-1].legend(fontsize=8);fig.savefig(FIG/'rms_drift.png',dpi=180);plt.close(fig)
errors=[]
for seed in [42,7,2026]:
    for b in range(2,5):
        a=np.load(OUT/f'baseline_seed{seed}'/f'fixed62_B{b}.npz')
        for ix in np.flatnonzero(a['truth']!=a['prediction']):
            row=features[b*510-510+int(ix)]
            errors.append(dict(seed=seed,block=f'B{b}',state=row['state'],window=int(row['window']),
                               start=int(row['start']),predicted=int(a['prediction'][ix])))
with (OUT/'error_windows.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=errors[0]);w.writeheader();w.writerows(errors)
fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
for ax,b in zip(axs,range(2,5)):
    for seed in [42,7,2026]:
        rr=[r for r in errors if r['seed']==seed and r['block']==f'B{b}']
        ax.scatter([r['window'] for r in rr],[['Health','Chipped','Miss','Root','Surface'].index(r['state'])+{42:-.15,7:0,2026:.15}[seed] for r in rr],label=str(seed),s=25)
    ax.set(title=f'Baseline fixed62 errors: B{b}',xlabel='Window index within state (0-101)',xlim=(-2,103),
           yticks=range(5),yticklabels=['Health','Chipped','Miss','Root','Surface']);ax.grid(alpha=.2)
axs[-1].legend(title='Seed');fig.savefig(FIG/'error_locations.png',dpi=180);plt.close(fig)
print('VERIFIED: 96 prediction/metric records; 12 baseline reproductions; wrote four figures.')
