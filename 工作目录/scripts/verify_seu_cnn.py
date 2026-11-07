"""从保存的模型重新预测入选配置，核对序列切片、标准化与预测。"""
from pathlib import Path
import argparse, json
import numpy as np
import torch
from train_seu_cnn import CNN, prepare

p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); args=p.parse_args()
root=args.root
winner=json.loads((root/'refined_ranking.json').read_text())[0]
data=np.load(root.parent/'development_signals.npz')
torch.set_num_threads(4)
device='cuda' if torch.cuda.is_available() else 'cpu'
for record in winner['folds']:
    c=record['configuration']; fold=record['fold']; seed=record['seed']
    d=root/'results'/f"{c['id']}_seed{seed}_fold{fold}"
    checkpoint=torch.load(d/'model.pt',map_location='cpu',weights_only=True)
    x,y,v,z,starts,mean,std=prepare(data,c['window'],fold)
    assert np.allclose(mean,checkpoint['mean']) and np.allclose(std,checkpoint['std'])
    model=CNN(c['channels'],c['kernel']).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    with torch.no_grad():
        probability=torch.cat([model(t).softmax(1) for t in torch.tensor(v,device=device).split(32)]).cpu().numpy()
    saved=np.load(d/'predictions.npz')
    assert np.array_equal(saved['truth'],z) and np.array_equal(saved['starts'],starts)
    assert np.array_equal(saved['prediction'],probability.argmax(1))
    assert np.allclose(saved['probability'],probability,atol=1e-5)
print('PASSED: selected configuration, all 9 saved models reproduce validation predictions')
(root/'verification.json').write_text(json.dumps({'status':'passed','reloaded_models':9,'configuration':winner['configuration']['id']}))
