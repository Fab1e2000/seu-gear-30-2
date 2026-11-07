"""导出前四个时间块的三轴信号；测试块不读取、不上传。"""
from pathlib import Path
from itertools import islice
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
cfg=json.loads((ROOT/'工作目录/configs/seu_30_2_split.json').read_text())
stop=max(cfg['blocks'][k]['stop'] for k in cfg['final_train_blocks'])
arrays={}
for source in cfg['source_files']:
    path=ROOT/source
    with path.open() as f:
        rows=islice(f,16,16+stop)
        x=np.loadtxt((line.replace('\\t','\t') for line in rows),usecols=(1,2,3))
    assert x.shape==(stop,3) and np.isfinite(x).all()
    arrays[path.stem.split('_')[0]]=x
    print(path.name,x.shape,flush=True)
out=ROOT/'工作目录/outputs/SEU-RF训练/development_signals.npz'
np.savez_compressed(out,**arrays)
print(out, out.stat().st_size,flush=True)
