"""Frozen basket-attachment feature experiment; --check verifies without model fitting."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
import hashlib
import importlib.metadata
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from colab_tabpfn import POOL,EVAL,TEST,RANK,CLASSES,BASELINE_GUARD,combine,log
from load_blocks import load_blocks

HERE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def probability(p,n):
    if p.shape!=(n,4) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-6):
        raise RuntimeError('invalid rank4 probabilities')
    return p


def read_inputs():
    manifest=json.loads((HERE/'basket_manifest.json').read_text())
    for name,digest in manifest.items():
        if sha(HERE/name)!=digest:raise RuntimeError(f'input hash mismatch: {name}')
    with (HERE/'panel_plain.pkl').open('rb') as f:panel=pickle.load(f)
    proposal=json.loads((HERE/'proposal_basket_attachment.json').read_text())
    if len(panel['feature_cols'])!=305:raise RuntimeError('expected baseline 305 features')
    if proposal['screen_gate']['origins']!=POOL or proposal['eval_gate']['origins']!=EVAL:
        raise RuntimeError('unexpected frozen origins')
    names=None
    for cutoff,block in panel['cutoffs'].items():
        extra,cols=load_blocks(HERE/'feature_layers',cutoff,block['ids'],['basket_attachment'])
        if names is not None and cols!=names:raise RuntimeError('feature schema differs by origin')
        names=cols
        if block['X'].dtype!=np.float64 or not np.isfinite(block['X']).all():
            raise RuntimeError('baseline precision/finite check failed')
        block['X_extended']=np.column_stack([block['X'],extra.astype(np.float64)])
        if block['X_extended'].shape!=(len(block['ids']),362):raise RuntimeError('expected 362 columns')
    if set(names)&set(panel['feature_cols']):raise RuntimeError('duplicate column names')
    panel['extended_cols']=panel['feature_cols']+names
    vector=np.asarray(json.loads((HERE/'pool_r4full_vector.json').read_text())['vR'],dtype=float)
    return panel,proposal,vector,manifest


def p17(cutoff):return np.load(HERE/'probs'/f'{cutoff}.npz',allow_pickle=False)['proba'].astype(float)

def base4(cutoff,n):return probability(np.load(HERE/'rank4'/f'{cutoff}.npz',allow_pickle=False)['proba'].astype(float),n)


def score(panel,cutoff,p,vector):
    pred=np.asarray(CLASSES)[np.argmax(combine(p17(cutoff),p)*vector,axis=1)]
    return float(f1_score(panel['cutoffs'][cutoff]['y'],pred,average='weighted',zero_division=0))


def shift(cutoff,months):return str(np.datetime64(cutoff,'M')+months)+'-01'


def split(panel,cutoff):
    latest=shift(cutoff,-3);limit=shift(latest,-3)
    origins=[o for o in panel['monthly_cutoffs'] if o<=latest and panel['cutoffs'][o]['y'] is not None]
    x=[];y=[];w=[];dates=[]
    for origin in origins:
        block=panel['cutoffs'][origin];labels=np.asarray(block['y'],dtype=str)
        collapsed=np.full(len(labels),'Other',dtype='<U10')
        collapsed[labels=='Inactivity']='Inactivity';collapsed[labels=='Stable']='Stable'
        collapsed[labels=='Existing-category growth: Fuel']='Fuel'
        year,month=map(int,origin[:7].split('-'));weight=.70+.05*((year-2024)*12+month-9)
        x.append(block['X_extended']);y.append(collapsed);w.append(np.full(len(labels),weight));dates.extend([origin]*len(labels))
    x=np.vstack(x);y=np.concatenate(y);w=np.concatenate(w);dates=np.asarray(dates)
    tr=dates<=limit;va=dates==latest
    if not tr.any() or not va.any() or not all(shift(o,3)<=latest for o in origins if o<=limit):
        raise RuntimeError('invalid purged inner split')
    if not all(shift(o,3)<=cutoff for o in origins):raise RuntimeError('outer training label leakage')
    return x,y,w,tr,va,{'origins':origins,'inner_validation_origin':latest,'inner_training_limit':limit,
                       'train_rows_full':len(y),'inner_training_rows':int(tr.sum()),'inner_validation_rows':int(va.sum())}


def fit_origin(panel,cutoff,proposal,identity):
    from lightgbm import LGBMClassifier,early_stopping
    block=panel['cutoffs'][cutoff];ids=block['ids'].astype(str)
    path=HERE/'basket_attachment_predictions'/f'{cutoff}.npz';path.parent.mkdir(exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            if str(data['identity'])!=identity or not np.array_equal(data['ids'],ids):
                raise RuntimeError(f'incompatible prediction cache: {path}; keep it and use a fresh run folder')
            return probability(data['proba'].astype(float),len(ids)),json.loads(str(data['metadata']))
    started=time.monotonic();x,y,w,tr,va,meta=split(panel,cutoff)
    def timeout_callback(env):
        if time.monotonic()-started>proposal['fit_timeout_seconds']:
            raise TimeoutError(f'{cutoff} exceeded the frozen per-origin fit budget')
    timeout_callback.order=5
    params=dict(proposal['head']);seeds=params.pop('seeds');patience=params.pop('early_stopping_rounds');minimum=params.pop('minimum_refit_rounds')
    total=np.zeros((len(ids),4));rounds_list=[];importance=np.zeros(x.shape[1])
    for seed in seeds:
        inner=LGBMClassifier(**params,random_state=seed)
        inner.fit(x[tr],y[tr],sample_weight=w[tr],eval_set=[(x[va],y[va])],
                  callbacks=[timeout_callback,early_stopping(patience,verbose=False)])
        rounds=max(int(inner.best_iteration_ or minimum),minimum);rounds_list.append(rounds)
        full=LGBMClassifier(**dict(params,n_estimators=rounds),random_state=seed)
        full.fit(x,y,sample_weight=w,callbacks=[timeout_callback])
        raw=full.booster_.predict(block['X_extended'])
        aligned=np.zeros((len(ids),4));indices={name:i for i,name in enumerate(full.classes_)}
        for j,name in enumerate(RANK):
            if name in indices:aligned[:,j]=raw[:,indices[name]]
        aligned/=aligned.sum(1,keepdims=True)
        total+=aligned;importance+=full.booster_.feature_importance(importance_type='gain')
        log(f'BASKET {cutoff} seed {seed}: rounds {rounds}, full rows {len(y)}, elapsed {time.monotonic()-started:.1f}s')
    p=probability(total/len(seeds),len(ids));importance/=len(seeds)
    meta.update({'rounds_by_seed':rounds_list,'features':x.shape[1],
                 'extra_gain_importance_share':float(importance[305:].sum()/max(importance.sum(),1e-12)),
                 'original_fit_seconds':round(time.monotonic()-started,1)})
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,ids=ids,proba=p,identity=identity,metadata=json.dumps(meta),importance=importance)
    temp.replace(path)
    return p,meta


def export(panel,p,vector,decision):
    source=HERE/'sasol_sprint16_r4full_vR.csv'
    pred=np.asarray(CLASSES)[np.argmax(combine(p17(TEST),p)*vector,axis=1)]
    with source.open(newline='') as f:rows=list(csv.reader(f))
    header,rows=rows[0],rows[1:]
    if header!=['ID','CLV_fuel','CLV_nonfuel','Opportunity']:raise RuntimeError('unexpected source columns')
    if len(rows)!=5488 or len({r[0] for r in rows})!=5488 or [r[0] for r in rows]!=panel['cutoffs'][TEST]['ids'].astype(str).tolist():
        raise RuntimeError('official IDs/order mismatch')
    clv=np.asarray([[float(r[1]),float(r[2])] for r in rows])
    if not np.isfinite(clv).all() or (clv<0).any():raise RuntimeError('invalid source CLV')
    target=HERE/'sasol_features_basket_attachment.csv';temp=target.with_suffix('.tmp')
    with temp.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows([r[:3]+[str(label)] for r,label in zip(rows,pred)])
    with temp.open(newline='') as f:check=list(csv.reader(f))[1:]
    if len(check)!=5488 or any(a[:3]!=b[:3] for a,b in zip(rows,check)):raise RuntimeError('ID/CLV source strings changed')
    temp.replace(target)
    decision['exported']={'file':target.name,'rows':len(rows),'changes_vs_best':sum(r[3]!=v for r,v in zip(rows,pred)),
                          'source_sha256':sha(source),'sha256':sha(target)}


def run_stages(panel,proposal,vector,baseline,decision,path,identity):
    def stage(origins,gate,name):
        result={'scores':{},'gains':{},'passed':False,'complete':False};decision[name]=result
        for cutoff in origins:
            p,meta=fit_origin(panel,cutoff,proposal,identity);decision['members'][cutoff]=meta
            value=score(panel,cutoff,p,vector);gain=value-baseline[cutoff]
            result['scores'][cutoff]=value;result['gains'][cutoff]=gain
            save(path,decision);log(f'BASKET-mix {cutoff}: {value:.6f}, gain {gain:+.6f}')
        result['mean_gain']=float(np.mean(list(result['gains'].values())));result['min_gain']=min(result['gains'].values())
        result['passed']=result['mean_gain']>=gate['mean_gain_min'] and result['min_gain']>=gate['per_origin_min'];result['complete']=True
        log(f'{name.upper()}: mean {result["mean_gain"]:+.6f}, min {result["min_gain"]:+.6f}, passed={result["passed"]}')
        save(path,decision);return result['passed']
    if stage(POOL,proposal['screen_gate'],'screen') and stage(EVAL,proposal['eval_gate'],'folds'):
        p,meta=fit_origin(panel,TEST,proposal,identity);decision['members'][TEST]=meta
        export(panel,p,vector,decision)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');args=parser.parse_args()
    panel,proposal,vector,manifest=read_inputs();baseline={}
    for cutoff in POOL+EVAL:
        block=panel['cutoffs'][cutoff]
        baseline[cutoff]=score(panel,cutoff,base4(cutoff,len(block['ids'])),vector)
        if abs(baseline[cutoff]-BASELINE_GUARD[cutoff])>1e-9:raise RuntimeError(f'baseline mismatch: {cutoff}')
        *_,meta=split(panel,cutoff)
        log(f'CHECK {cutoff}: baseline {baseline[cutoff]:.6f}, features 362, full rows {meta["train_rows_full"]}, purged inner {meta["inner_training_rows"]}/{meta["inner_validation_rows"]}')
    if args.check:
        log('PASS: input hashes, all feature IDs/order, float64 baseline precision, seven baseline guards and purged splits; no fits run');return
    versions={name:importlib.metadata.version(name) for name in ['numpy','scikit-learn','lightgbm']}
    if versions!=proposal['versions']:raise RuntimeError(f'Install the pinned package versions: expected {proposal["versions"]}, got {versions}')
    identity=hashlib.sha256((json.dumps(manifest,sort_keys=True)+json.dumps(versions,sort_keys=True)).encode()).hexdigest()
    decision={'experiment':proposal['name'],'proposal_sha256':sha(HERE/'proposal_basket_attachment.json'),
              'identity':identity,'versions':versions,'python':platform.python_version(),'device':'cpu',
              'baseline':baseline,'members':{},'comparison':'Frozen archived baseline probabilities; original float64 305-column inputs and production rank4 recipe; only basket-attachment feature block added.'}
    path=HERE/'basket_attachment_decision.json';save(path,decision);started=time.monotonic()
    try:run_stages(panel,proposal,vector,baseline,decision,path,identity)
    except TimeoutError as exc:decision['aborted']=str(exc)
    except Exception as exc:
        decision['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        decision['this_run_seconds']=round(time.monotonic()-started,1)
        decision['original_member_fit_seconds_total']=round(sum(m['original_fit_seconds'] for m in decision['members'].values()),1)
        save(path,decision)
    print('\n===== COPY THIS BACK =====');print(json.dumps(decision,indent=2))
    print('Send back basket_attachment_decision.json and basket_attachment_predictions/*.npz')


if __name__=='__main__':main()
