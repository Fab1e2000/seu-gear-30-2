from pathlib import Path
import json, time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
from train_seu_rf_expanded import extract,NAMES,STATES
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=True)
parent=root.parent/'expanded-grid'
search=json.loads((parent/'seu_30_2_rf_expanded.json').read_text())
best=json.loads((parent/'results/summary.json').read_text())['top_configuration']
d=np.load(parent/'results/development_features.npz')
assert list(d['feature_names'])==NAMES and d['X'].shape==(2040,24)
assert best['id']=='EXP0721' and best['feature_set']=='C'
model=RandomForestClassifier(**search['fixed_parameters'],**best['parameters'],n_jobs=8)
t=time.time();model.fit(d['X'],d['y']);joblib.dump({'model':model,'feature_names':NAMES,'states':STATES,'configuration':best},out/'random_forest.joblib')
raw=np.load(root/'test_signals.npz');xs=[];ys=[]
for i,state in enumerate(STATES):
 x=raw[state]; assert x.shape==(209712,3)
 xs.append(extract(x[:102*2048].reshape(102,2048,3)));ys.extend([i]*102)
X=np.concatenate(xs);y=np.array(ys)
loaded=joblib.load(out/'random_forest.joblib')['model']; pred=loaded.predict(X)
assert np.array_equal(pred,model.predict(X))
np.savez_compressed(out/'test_predictions.npz',X=X,y=y,prediction=pred,probabilities=loaded.predict_proba(X),feature_names=NAMES)
s={'configuration':best,'train_windows':2040,'test_windows':510,'test_evaluated':True,'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'confusion_matrix':confusion_matrix(y,pred,labels=range(5)).tolist(),'classification_report':classification_report(y,pred,labels=range(5),target_names=STATES,output_dict=True),'elapsed_seconds':time.time()-t}
(out/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');print(json.dumps(s),flush=True)
