"""在远端运行 108 套配置的三折时间验证，仅使用开发段。"""
from pathlib import Path
import argparse, csv, json, os, platform, time
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

FEATURES=['mean','std','rms','peak','peak_to_peak','skewness','kurtosis','crest_factor']
STATES=['Health','Chipped','Miss','Root','Surface']
NAMES=[f'{a}_{f}' for a in 'xyz' for f in FEATURES]


def extract(x):
    mean=x.mean(axis=1); std=x.std(axis=1)
    rms=np.sqrt((x*x).mean(axis=1)); peak=np.abs(x).max(axis=1)
    assert (std>0).all() and (rms>0).all()
    z=(x-mean[:,None,:])/std[:,None,:]
    values=[mean,std,rms,peak,np.ptp(x,axis=1),(z**3).mean(axis=1),(z**4).mean(axis=1),peak/rms]
    result=np.stack(values,axis=2).reshape(len(x),24)
    assert np.isfinite(result).all()
    assert np.allclose(rms*rms,std*std+mean*mean)
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args(); root=args.root
    split=json.loads((root/'seu_30_2_split.json').read_text(encoding='utf-8'))
    search=json.loads((root/'seu_30_2_rf_candidates.json').read_text(encoding='utf-8'))
    out=root/'results'; out.mkdir(exist_ok=True)
    started=time.time()
    metadata={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'host':platform.node(),'platform':platform.platform(),'cpu_count':os.cpu_count(),'random_state':42,'test_evaluated':False,'states':STATES}
    (out/'environment.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    data=np.load(root/'development_signals.npz')
    xs=[]; ys=[]; bs=[]; starts=[]
    size=split['window_size']; end=max(split['blocks'][k]['stop'] for k in split['final_train_blocks'])
    assert end==split['blocks'][split['test_blocks'][0]]['start']
    for label,state in enumerate(STATES):
        raw=data[state]; assert raw.shape==(end,3)
        for key in split['final_train_blocks']:
            b=split['blocks'][key]; n=(b['stop']-b['start'])//size
            x=raw[b['start']:b['start']+n*size].reshape(n,size,3)
            xs.append(extract(x)); ys.extend([label]*n); bs.extend([key]*n)
            starts.extend(b['start']+np.arange(n)*size)
    X=np.concatenate(xs); y=np.array(ys); blocks=np.array(bs)
    assert X.shape==(2040,24)
    np.savez_compressed(out/'development_features.npz',X=X,y=y,blocks=blocks,starts=starts,feature_names=NAMES)
    folds=[]
    for i,f in enumerate(split['folds']):
        tr=np.flatnonzero(np.isin(blocks,f['train_blocks'])); va=np.flatnonzero(np.isin(blocks,f['validation_blocks']))
        assert not np.intersect1d(tr,va).size and len(tr)==510*(i+1) and len(va)==510
        assert np.bincount(y[tr]).tolist()==[102*(i+1)]*5
        assert np.bincount(y[va]).tolist()==[102]*5
        folds.append((tr,va))
    records=[]; predictions=[]
    with (out/'fold_results.jsonl').open('w',encoding='utf-8') as stream:
        for ci,c in enumerate(search['configurations'],1):
            cols=[NAMES.index(k) for k in search['feature_sets'][c['feature_set']]]
            scores=[]
            for fi,(tr,va) in enumerate(folds,1):
                model=RandomForestClassifier(**search['fixed_parameters'],**c['parameters'],n_jobs=min(8,os.cpu_count() or 1))
                model.fit(X[tr][:,cols],y[tr]); pred=model.predict(X[va][:,cols])
                score=f1_score(y[va],pred,labels=range(5),average='macro',zero_division=0); scores.append(float(score))
                row={'id':c['id'],'feature_set':c['feature_set'],'fold':fi,'macro_f1':float(score),'accuracy':float(accuracy_score(y[va],pred)),'confusion_matrix':confusion_matrix(y[va],pred,labels=range(5)).tolist()}
                stream.write(json.dumps(row)+'\n'); stream.flush(); predictions.append(pred)
                print(f"{c['id']} fold={fi}/3 macro_f1={score:.6f} fits={(ci-1)*3+fi}/324",flush=True)
            records.append({**c,'fold_macro_f1':scores,'mean_macro_f1':float(np.mean(scores)),'std_macro_f1':float(np.std(scores))})
    ranking=sorted(records,key=lambda r:(-r['mean_macro_f1'],r['id']))
    (out/'ranking.json').write_text(json.dumps(ranking,indent=2),encoding='utf-8')
    with (out/'ranking.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=['id','feature_set','n_estimators','max_depth','min_samples_leaf','max_features','fold1','fold2','fold3','mean_macro_f1','std_macro_f1']); writer.writeheader()
        for r in ranking:
            writer.writerow({'id':r['id'],'feature_set':r['feature_set'],**r['parameters'],**{f'fold{i+1}':v for i,v in enumerate(r['fold_macro_f1'])},'mean_macro_f1':r['mean_macro_f1'],'std_macro_f1':r['std_macro_f1']})
    np.savez_compressed(out/'validation_predictions.npz',predictions=np.array(predictions),validation_indices=np.array([v for _,v in folds]),states=STATES)
    summary={'status':'completed','configurations':len(records),'fits':len(predictions),'elapsed_seconds':time.time()-started,'test_evaluated':False,'best_per_feature_set':{k:next(r for r in ranking if r['feature_set']==k) for k in search['feature_sets']},'top_configuration':ranking[0]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
