"""使用开发侧确定的配置与轮数训练，再评价最后一个时间块。"""
from pathlib import Path
import json, random, platform, time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from train_seu_cnn import CNN, STATES, BLOCK, save

root=Path(__file__).resolve().parent
out=root/'results'; out.mkdir(exist_ok=True)
winner=json.loads((root.parent/'cnn-search/refined_ranking.json').read_text())[0]
c=winner['configuration']; assert c['id']=='CNN023'
epoch_values=[r['best_epoch'] for r in winner['folds']]
epochs=int(np.median(epoch_values)); seed=42
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
torch.set_num_threads(4); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
device='cuda' if torch.cuda.is_available() else 'cpu'
data=np.load(root.parent/'development_signals.npz')
raw=np.concatenate([data[s] for s in STATES]); mean=raw.mean(0);std=raw.std(0)
assert (std>0).all()
w=c['window'];n=BLOCK//w
xs=[];ys=[]
for label,s in enumerate(STATES):
    assert data[s].shape==(4*BLOCK,3)
    for b in range(4):
        x=data[s][b*BLOCK:b*BLOCK+n*w].reshape(n,w,3)
        xs.append(((x-mean)/std).transpose(0,2,1).astype('float32'));ys.extend([label]*n)
x=torch.tensor(np.concatenate(xs),device=device); y=torch.tensor(ys,device=device)
assert len(y)==2040 and torch.isfinite(x).all()
model=CNN(c['channels'],c['kernel']).to(device)
optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate']);criterion=nn.CrossEntropyLoss()
plan=dict(configuration=c,epochs=epochs,epoch_rule='median of nine CV best epochs',cv_best_epochs=epoch_values,seed=seed,train_windows=len(y),test_previously_viewed=True)
save(out/'training_plan.json',plan)
history=[];began=time.time()
for epoch in range(1,epochs+1):
    model.train(); total=torch.zeros((),device=device)
    for indices in torch.randperm(len(y),device=device).split(32):
        optimizer.zero_grad(set_to_none=True);loss=criterion(model(x[indices]),y[indices]);loss.backward();optimizer.step();total+=loss.detach()*len(indices)
    value=float(total/len(y));assert np.isfinite(value)
    history.append(dict(epoch=epoch,train_loss=value))
    print(f'epoch={epoch}/{epochs} train_loss={value:.6f}',flush=True)
training_seconds=time.time()-began
save(out/'history.json',history)
checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},configuration=c,mean=mean.tolist(),std=std.tolist(),epochs=epochs,seed=seed,states=STATES)
torch.save(checkpoint,out/'model.pt')
# Test segment is first opened after the final model has been saved.
test=np.load(root.parent/'final/test_signals.npz');xs=[];truth=[]
for label,s in enumerate(STATES):
    assert test[s].shape==(BLOCK,3)
    xx=test[s][:n*w].reshape(n,w,3)
    xs.append(((xx-mean)/std).transpose(0,2,1).astype('float32'));truth.extend([label]*n)
v=torch.tensor(np.concatenate(xs),device=device);truth=np.array(truth)
assert len(truth)==510 and torch.isfinite(v).all()
model.eval()
with torch.no_grad(): expected=torch.cat([model(batch).softmax(1) for batch in v.split(32)]).cpu().numpy()
loaded=torch.load(out/'model.pt',map_location='cpu',weights_only=True)
restored=CNN(c['channels'],c['kernel']).to(device);restored.load_state_dict(loaded['state_dict']);restored.eval()
with torch.no_grad():probability=torch.cat([restored(batch).softmax(1) for batch in v.split(32)]).cpu().numpy()
assert np.allclose(expected,probability,atol=1e-7)
pred=probability.argmax(1);cm=confusion_matrix(truth,pred,labels=range(5))
score=float(f1_score(truth,pred,average='macro'))
assert np.isclose(score,np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1))))
assert cm.sum(1).tolist()==[102]*5
np.savez_compressed(out/'test_predictions.npz',truth=truth,prediction=pred,probability=probability,starts=np.tile(4*BLOCK+np.arange(n)*w,5),states=STATES)
summary=dict(**plan,test_windows=len(truth),accuracy=float(accuracy_score(truth,pred)),macro_f1=score,confusion_matrix=cm.tolist(),classification_report=classification_report(truth,pred,target_names=STATES,output_dict=True,zero_division=0),training_seconds=training_seconds,test_evaluated=True,reload_verified=True)
save(out/'summary.json',summary)
save(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None))
print(json.dumps(summary),flush=True)
