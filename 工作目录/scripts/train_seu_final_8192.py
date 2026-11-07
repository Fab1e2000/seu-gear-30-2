from pathlib import Path
import json, time, platform, sklearn
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
from train_seu_rf_8192 import extract,NAMES,STATES
root=Path(__file__).resolve().parent
out=root/'results';out.mkdir(exist_ok=True)
parent=root.parent/'window8192-simple'
search=json.loads((parent/'seu_30_2_rf_candidates.json').read_text())
best=json.loads((parent/'results/summary.json').read_text())['top_configuration']
d=np.load(parent/'results/development_features.npz')
assert list(d['feature_names'])==NAMES and d['X'].shape==(500,24)
assert best['id']=='RF097' and best['feature_set']=='C'
model=RandomForestClassifier(**search['fixed_parameters'],**best['parameters'],n_jobs=8)
t=time.time();model.fit(d['X'],d['y']);joblib.dump({'model':model,'feature_names':NAMES,'states':STATES,'configuration':best},out/'random_forest.joblib')
raw=np.load(root.parent/'final/test_signals.npz');xs=[];ys=[]
for i,state in enumerate(STATES):
 x=raw[state]; assert x.shape==(209712,3)
 xs.append(extract(x[:25*8192].reshape(25,8192,3)));ys.extend([i]*25)
X=np.concatenate(xs);y=np.array(ys)
loaded=joblib.load(out/'random_forest.joblib')['model']; pred=loaded.predict(X)
assert np.array_equal(pred,model.predict(X))
np.savez_compressed(out/'test_predictions.npz',X=X,y=y,prediction=pred,probabilities=loaded.predict_proba(X),feature_names=NAMES)
s={'configuration':best,'train_windows':500,'test_windows':125,'test_evaluated':True,'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),'confusion_matrix':confusion_matrix(y,pred,labels=range(5)).tolist(),'classification_report':classification_report(y,pred,labels=range(5),target_names=STATES,output_dict=True),'elapsed_seconds':time.time()-t}
(out/'summary.json').write_text(json.dumps(s,indent=2),encoding='utf-8');print(json.dumps(s),flush=True)

cm=np.array(s['confusion_matrix'])
assert cm.sum(axis=1).tolist()==[25]*5
assert abs(np.trace(cm)/cm.sum()-s['accuracy'])<1e-12
assert abs(np.mean(2*np.diag(cm)/(cm.sum(0)+cm.sum(1)))-s['macro_f1'])<1e-12
(out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'host':platform.node(),'parameters':model.get_params(),'window_size':8192,'test_previously_viewed':True},indent=2),encoding='utf-8')
