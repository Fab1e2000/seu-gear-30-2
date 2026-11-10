"""CNN 时间泛化诊断：预设四组配置，固定轮数与 B2 早停对照。"""
import argparse
import csv
import platform
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from train_seu_cnn import BLOCK, CNN, STATES, save

ARMS = [dict(name='baseline', channels=[16,32,64], normalization='train_axis', weight_decay=0.),
        dict(name='window_norm', channels=[16,32,64], normalization='window_axis', weight_decay=0.),
        dict(name='small', channels=[8,16,32], normalization='train_axis', weight_decay=0.),
        dict(name='weight_decay', channels=[16,32,64], normalization='train_axis', weight_decay=1e-4)]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    device='cuda' if torch.cuda.is_available() else 'cpu'
    w=2048; n=BLOCK//w; seeds=[42,7,2026]
    data=np.load(args.data)
    assert all(data[s].shape==(4*BLOCK,3) for s in STATES)
    raw=[np.stack([data[s][b*BLOCK:b*BLOCK+n*w].reshape(n,w,3) for s in STATES]) for b in range(4)]
    train=raw[0].reshape(-1,3); mean=train.mean(0); std=train.std(0)
    truth=np.repeat(np.arange(5),n); y=torch.tensor(truth,device=device)
    plan=dict(arms=ARMS,seeds=seeds,window=w,kernel=17,learning_rate=.001,batch_size=32,
              fixed_epochs=62,max_epochs=100,patience=15,training='B1',early_stopping='B2 loss',
              evaluation=['B3','B4'],test_B5_loaded=False,normalization_fit='actual B1 windows',
              design='development sensitivity analysis; all blocks previously used in selection',
              device=device,torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
              gpu=torch.cuda.get_device_name(0) if device=='cuda' else None)
    save(args.out/'plan.json',plan)
    feature_rows=[]
    for b,a in enumerate(raw):
        for label,s in enumerate(STATES):
            for j,x in enumerate(a[label]):
                row=dict(block=f'B{b+1}',state=s,label=label,window=j,start=b*BLOCK+j*w)
                for axis,k in enumerate('xyz'):
                    row[k+'_mean']=float(x[:,axis].mean())
                    row[k+'_rms']=float(np.sqrt(np.mean(x[:,axis]**2)))
                    row[k+'_std']=float(x[:,axis].std())
                feature_rows.append(row)
    with (args.out/'window_features.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=feature_rows[0]);writer.writeheader();writer.writerows(feature_rows)
    criterion=nn.CrossEntropyLoss(); results=[]
    for arm in ARMS:
        blocks=[]
        for a in raw:
            a=a.reshape(-1,w,3)
            if arm['normalization']=='window_axis':
                scale=a.std(1,keepdims=True); assert (scale>0).all()
                a=(a-a.mean(1,keepdims=True))/scale
            else: a=(a-mean)/std
            assert np.isfinite(a).all()
            blocks.append(torch.tensor(a.transpose(0,2,1).astype('float32'),device=device))
        for seed in seeds:
            folder=args.out/f"{arm['name']}_seed{seed}";folder.mkdir(exist_ok=True)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            model=CNN(arm['channels'],17).to(device)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=arm['weight_decay'])
            history=[];best=float('inf');best_epoch=0;stop_epoch=None;best_state=None;fixed_state=None
            def evaluate(b):
                model.eval()
                with torch.no_grad():
                    logits=torch.cat([model(x) for x in blocks[b].split(32)])
                    return float(criterion(logits,y)),logits.softmax(1).cpu().numpy()
            for epoch in range(1,101):
                model.train();total=0.
                for ix in torch.randperm(len(y),device=device).split(32):
                    opt.zero_grad(set_to_none=True);loss=criterion(model(blocks[0][ix]),y[ix])
                    loss.backward();opt.step();total+=float(loss.detach())*len(ix)
                train_loss,_=evaluate(0);val_loss,_=evaluate(1)
                history.append(dict(epoch=epoch,online_train_loss=total/len(y),train_loss=train_loss,validation_loss=val_loss))
                if stop_epoch is None:
                    if val_loss<best:
                        best=val_loss;best_epoch=epoch
                        best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    if epoch-best_epoch>=15: stop_epoch=epoch
                if epoch==62: fixed_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                if stop_epoch is not None and epoch>=62: break
            save(folder/'history.json',history)
            assert fixed_state is not None and best_state is not None
            for policy,state,selected_epoch in [('fixed62',fixed_state,62),('early',best_state,best_epoch)]:
                model.load_state_dict(state)
                torch.save(dict(state_dict=state,arm=arm,epoch=selected_epoch,mean=mean.tolist(),std=std.tolist()),folder/f'{policy}.pt')
                for b in range(4):
                    loss,prob=evaluate(b);pred=prob.argmax(1)
                    cm=confusion_matrix(truth,pred,labels=range(5))
                    row=dict(arm=arm['name'],seed=seed,policy=policy,epoch=selected_epoch,
                             stop_epoch=stop_epoch,block=f'B{b+1}',loss=loss,
                             accuracy=float((pred==truth).mean()),macro_f1=float(f1_score(truth,pred,average='macro')),
                             errors=int((pred!=truth).sum()),confusion_matrix=cm.tolist(),
                             parameters=sum(p.numel() for p in model.parameters()))
                    results.append(row)
                    np.savez_compressed(folder/f'{policy}_B{b+1}.npz',truth=truth,prediction=pred,probability=prob)
            save(args.out/'summary.json',dict(plan=plan,results=results))
            print(f"{arm['name']} seed={seed} done; best={best_epoch}; stop={stop_epoch}",flush=True)
    assert len(results)==96

if __name__=='__main__':
    main()
