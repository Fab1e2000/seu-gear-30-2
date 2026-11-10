"""固定 B1 训练，评价 B2/B3/B4；沿用 Windows CNN 环境，不读取 B5。"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from train_seu_cnn import BLOCK, CNN, STATES, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    window, epochs = 2048, 62
    count = BLOCK // window
    data = np.load(args.data)
    assert all(data[s].shape == (4 * BLOCK, 3) for s in STATES)
    raw = np.concatenate([data[s][:count * window] for s in STATES])
    mean, std = raw.mean(0), raw.std(0)
    assert (std > 0).all()
    truth = np.repeat(np.arange(5), count)
    blocks = []
    for block in range(4):
        x = np.concatenate([data[s][block * BLOCK:block * BLOCK + count * window]
                            .reshape(count, window, 3) for s in STATES])
        x = ((x - mean) / std).transpose(0, 2, 1).astype('float32')
        assert np.isfinite(x).all()
        blocks.append(torch.tensor(x, device=device))
    y = torch.tensor(truth, device=device)
    plan = dict(configuration='CNN023',window=window,epochs=epochs,seeds=[42,7,2026],
                train_windows=len(truth),evaluation_windows_per_block=len(truth),
                train_range=[0,count*window],
                gaps_samples=[b*BLOCK-count*window for b in [1,2,3]],
                epoch_rule='fixed before this experiment from previous development selection',
                selection_on_new_scores=False,test_block_loaded=False,
                device=device,torch=torch.__version__,mean=mean.tolist(),std=std.tolist())
    save(args.out/'plan.json',plan)
    rows=[]
    for seed in plan['seeds']:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model=CNN([16,32,64],17).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        criterion=nn.CrossEntropyLoss()
        history=[]
        for epoch in range(1,epochs+1):
            model.train(); total=0.
            for ix in torch.randperm(len(y),device=device).split(32):
                optimizer.zero_grad(set_to_none=True)
                loss=criterion(model(blocks[0][ix]),y[ix])
                loss.backward(); optimizer.step()
                total+=float(loss.detach())*len(ix)
            history.append(dict(epoch=epoch,train_loss=total/len(y)))
        save(args.out/f'history_seed{seed}.json',history)
        torch.save(dict(state_dict=model.cpu().state_dict(),mean=mean,std=std),
                   args.out/f'model_seed{seed}.pt')
        model.to(device).eval()
        for b,x in enumerate(blocks):
            with torch.no_grad():
                logits=torch.cat([model(batch) for batch in x.split(32)])
                loss=float(criterion(logits,y))
                probability=logits.softmax(1).cpu().numpy()
            prediction=probability.argmax(1)
            cm=confusion_matrix(truth,prediction,labels=range(5))
            assert cm.sum()==len(truth)
            row=dict(seed=seed,block=f'B{b+1}',loss=loss,
                     accuracy=float(accuracy_score(truth,prediction)),
                     macro_f1=float(f1_score(truth,prediction,average='macro')),
                     errors=int((prediction!=truth).sum()),confusion_matrix=cm.tolist())
            rows.append(row)
            np.savez_compressed(args.out/f'predictions_seed{seed}_B{b+1}.npz',
                                truth=truth,prediction=prediction,probability=probability)
            print(json.dumps(row),flush=True)
    save(args.out/'summary.json',dict(plan=plan,results=rows))


if __name__=='__main__':
    main()
