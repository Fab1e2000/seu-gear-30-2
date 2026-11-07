"""SEU 三轴 CNN 时间交叉验证；仅加载前 80% 信号。"""
import argparse, copy, itertools, json, os, platform, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

STATES=['Health','Chipped','Miss','Root','Surface']
BLOCK=209712

def save(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)

class CNN(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        layers=[]; previous=3
        for i,c in enumerate(channels):
            layers.extend([nn.Conv1d(previous,c,kernel,padding=kernel//2),nn.ReLU()])
            if i<2: layers.append(nn.MaxPool1d(2))
            previous=c
        self.features=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Linear(previous,5)
    def forward(self,x):
        return self.head(self.pool(self.features(x)).squeeze(-1))

def configurations():
    return [dict(id=f'CNN{i:03}',window=w,kernel=k,channels=list(c),learning_rate=lr)
            for i,(w,k,c,lr) in enumerate(itertools.product([1024,2048,4096,8192],[5,9,17],[(8,16,32),(16,32,64)],[.001,.0003]),1)]

def prepare(data, window, fold):
    # Statistics include all classes and only raw training blocks, separately per axis.
    raw=np.concatenate([data[s][:BLOCK*fold] for s in STATES])
    mean=raw.mean(0); std=raw.std(0)
    assert (std>0).all()
    xs=[]; ys=[]; bs=[]; starts=[]
    n=BLOCK//window
    for label,s in enumerate(STATES):
        for b in range(fold+1):
            start=b*BLOCK
            x=data[s][start:start+n*window].reshape(n,window,3)
            xs.append(((x-mean)/std).transpose(0,2,1).astype('float32'))
            ys.extend([label]*n); bs.extend([b]*n); starts.extend(start+np.arange(n)*window)
    x=np.concatenate(xs); y=np.array(ys); b=np.array(bs); starts=np.array(starts)
    tr=b<fold; va=b==fold
    assert np.bincount(y[va]).tolist()==[n]*5
    assert starts[tr].max()+window<=fold*BLOCK<=starts[va].min()
    assert np.isfinite(x).all()
    return x[tr],y[tr],x[va],y[va],starts[va],mean,std

def train(root, data, c, fold, seed, device, smoke=False):
    directory=root/'results'/f"{c['id']}_seed{seed}_fold{fold}"
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'metrics.json').exists():
        return json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    x,y,v,z,starts,mean,std=prepare(data,c['window'],fold)
    x=torch.tensor(x,device=device); y=torch.tensor(y,device=device)
    v=torch.tensor(v,device=device); z=torch.tensor(z,device=device)
    model=CNN(c['channels'],c['kernel']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    criterion=nn.CrossEntropyLoss()
    best=float('inf'); best_epoch=0; best_state=None; history=[]; began=time.time()
    for epoch in range(1,(2 if smoke else 100)+1):
        model.train(); order=torch.randperm(len(y),device=device); loss_total=torch.zeros((),device=device)
        for inds in order.split(32):
            optimizer.zero_grad(set_to_none=True)
            loss=criterion(model(x[inds]),y[inds]); loss.backward(); optimizer.step()
            loss_total+=loss.detach()*len(inds)
        model.eval()
        with torch.no_grad():
            logits=torch.cat([model(batch) for batch in v.split(32)])
            vl=float(criterion(logits,z)); tl=float(loss_total/len(y))
        assert np.isfinite([vl,tl]).all()
        history.append(dict(epoch=epoch,train_loss=tl,validation_loss=vl))
        if vl<best:
            best=vl; best_epoch=epoch
            best_state={k:t.detach().cpu().clone() for k,t in model.state_dict().items()}
        if epoch-best_epoch>=15: break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        probability=torch.cat([model(batch).softmax(1) for batch in v.split(32)]).cpu().numpy()
    truth=z.cpu().numpy(); prediction=probability.argmax(1)
    cm=confusion_matrix(truth,prediction,labels=range(5))
    score=float(f1_score(truth,prediction,labels=range(5),average='macro',zero_division=0))
    denom=cm.sum(0)+cm.sum(1)
    assert np.isclose(score,np.mean(np.divide(2*np.diag(cm),denom,out=np.zeros(5),where=denom!=0)))
    torch.save(dict(state_dict=best_state,configuration=c,mean=mean.tolist(),std=std.tolist(),seed=seed,fold=fold,states=STATES),directory/'model.pt')
    np.savez_compressed(directory/'predictions.npz',truth=truth,prediction=prediction,probability=probability,starts=starts)
    save(directory/'history.json',history)
    result=dict(configuration=c,fold=fold,seed=seed,train_windows=len(y),validation_windows=len(z),epochs=len(history),best_epoch=best_epoch,validation_loss=best,macro_f1=score,accuracy=float(accuracy_score(truth,prediction)),confusion_matrix=cm.tolist(),seconds=time.time()-began,parameters=sum(p.numel() for p in model.parameters()))
    save(directory/'metrics.json',result)
    print(f"{c['id']} seed={seed} fold={fold} epochs={len(history)} F1={score:.6f} seconds={result['seconds']:.1f}",flush=True)
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); p.add_argument('--smoke',action='store_true'); args=p.parse_args()
    root=args.root; root.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    config=configurations(); assert len(config)==48
    save(root/'configurations.json',dict(configurations=config,fixed=dict(optimizer='Adam',batch_size=32,max_epochs=100,patience=15,loss='CrossEntropyLoss',weight_decay=0,dropout=0,seed=42,refinement_seeds=[7,2026],refinement_selection="all_tied_first"),test_evaluated=False))
    save(root/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,host=platform.node()))
    data=np.load(root.parent/'development_signals.npz')
    assert all(data[s].shape==(4*BLOCK,3) for s in STATES)
    if args.smoke:
        train(root/'smoke',data,config[0],1,42,device,True)
        for w in [1024,2048,4096,8192]:
            with torch.no_grad(): assert CNN([16,32,64],17)(torch.zeros(2,3,w)).shape==(2,5)
        print('SMOKE PASSED',flush=True); return
    rows=[]
    for c in config:
        records=[train(root,data,c,f,42,device) for f in [1,2,3]]
        scores=[r['macro_f1'] for r in records]
        rows.append(dict(configuration=c,folds=records,mean_macro_f1=float(np.mean(scores)),std_macro_f1=float(np.std(scores))))
        save(root/'progress.json',dict(stage='search',completed_configurations=len(rows),total_configurations=48))
    ranking=sorted(rows,key=lambda r:(-r['mean_macro_f1'],r['configuration']['id']))
    save(root/'ranking.json',ranking)
    refined=[]
    finalists=[r for r in ranking if r['mean_macro_f1']==ranking[0]['mean_macro_f1']]
    for row in finalists:
        c=row['configuration']; records=list(row['folds'])
        for seed in [7,2026]:
            records.extend(train(root,data,c,f,seed,device) for f in [1,2,3])
        scores=[r['macro_f1'] for r in records]
        refined.append(dict(configuration=c,folds=records,mean_macro_f1=float(np.mean(scores)),std_macro_f1=float(np.std(scores))))
    refined.sort(key=lambda r:(-r['mean_macro_f1'],r['configuration']['id']))
    save(root/'refined_ranking.json',refined)
    save(root/'summary.json',dict(status='completed',search_fits=144,refinement_fits=len(finalists)*6,total_fits=144+len(finalists)*6,test_evaluated=False,winner=refined[0]))
    save(root/'progress.json',dict(stage='completed',fits=144+len(finalists)*6))
    print(f'COMPLETED {144+len(finalists)*6} fits',flush=True)

if __name__=='__main__': main()
