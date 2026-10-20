"""Three fixed model families, six declared prediction views; no adaptive search."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,csv,gc,hashlib,importlib.metadata,json,pickle,subprocess,sys,time
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from colab_tabpfn import CLASSES,POOL,EVAL,TEST,BASELINE_GUARD,combine
from load_blocks import load_blocks

HERE=Path(__file__).resolve().parent
ARMS=['memory17','balanced17','rank17']
BLOCKS=['shop_cadence','basket_attachment','category_state','coverage_and_ratios','fine_category_and_site']
CONTEXT=['log_n_tx','n_locations','primary_share','recency_days','age_days','gap_mean','gap_cv',
         'fuel_recency','shop_recency','fuel_tx','shop_tx','fuel_z_q1','shop_z_q1','fuel_z_q2','shop_z_q2',
         'fuel_dz','shop_dz','fuel_growth_gap','n_cat_ever','n_cat_q1','origin_month']
FIELDS=['ever','before_q1','q1_purchase_baskets','q2_purchase_baskets','q1_net_rand','q2_net_rand',
        'q1_purchase_days','q2_purchase_days','recency_days','first_observed_recent','reactivated',
        'became_dormant','signed_log_spend_change','growth_absolute_spend_requirement','growth_relative_spend_boundary']


def log(s):print(time.strftime('%H:%M:%S'),s,flush=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def shift(c,m):return str(np.datetime64(c,'M')+m)+'-01'
def slug(s):return s.lower().replace('&','and').replace(' ','_').replace('-','_').replace('/','_').replace(':','')
def valid(p,n):
    if p.shape!=(n,17) or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('invalid class probabilities')
    return p


def choice_features(base,base_names,extra,names,config):
    """17 candidate rows per customer; class labels are never used as features."""
    pos={c:i for i,c in enumerate(base_names)};epos={c:i for i,c in enumerate(names)}
    context=CONTEXT+[c for c in base_names if c.startswith('prev__')]
    n=len(base);parts=[np.repeat(base[:,[pos[c] for c in context]].astype(np.float32),17,axis=0)]
    label=np.tile(np.arange(17,dtype=np.float32),n)
    specific=np.zeros((n,17,len(FIELDS)+3),dtype=np.float32)
    for j,label_name in enumerate(CLASSES):
        role=0 if label_name=='Inactivity' else 1 if label_name=='Stable' else 2 if label_name.startswith('Existing-category growth:') else 3
        specific[:,j,-3]=role
        key='prev__'+slug(label_name)
        if key in pos:specific[:,j,-2]=base[:,pos[key]]
        if role<2:
            specific[:,j,-1]=1;continue
        kind,category=label_name.split(': ',1)
        cats=[c for c,v in config['category_mapping'].items() if v['included'] and v[kind]==category]
        if not cats:raise RuntimeError(f'no original categories for {label_name}')
        for k,field in enumerate(FIELDS):
            values=np.column_stack([extra[:,epos['cs_'+slug(c)+'_'+field]] for c in cats])
            specific[:,j,k]=values.max(1)
        ever=np.column_stack([extra[:,epos['cs_'+slug(c)+'_ever']] for c in cats])
        specific[:,j,-1]=(ever.max(1)>0) if role==2 else (ever.min(1)==0)
    parts.extend([label[:,None],specific.reshape(n*17,-1)])
    x=np.column_stack(parts)
    if not np.isfinite(x).all():raise ValueError('nonfinite choice features')
    return x,len(context)


def inputs():
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,h in manifest.items():
        if sha(HERE/name)!=h:raise RuntimeError('input hash mismatch '+name)
    proposal=json.loads((HERE/'proposal.json').read_text());config=json.loads((HERE/'label_config.json').read_text())
    with (HERE/'panel_plain.pkl').open('rb') as f:panel=pickle.load(f)
    for cutoff,b in panel['cutoffs'].items():
        extra,names=load_blocks(HERE/'feature_layers',cutoff,b['ids'],BLOCKS)
        if extra.shape[1]!=489:raise RuntimeError('expected all 489 candidate features')
        base=b.pop('X');b['rank_x'],b['rank_label_column']=choice_features(base,panel['feature_cols'],extra,names,config)
        b['X']=np.column_stack([base.astype(np.float32),extra])
        if b['X'].shape[1]!=794:raise RuntimeError('expected 794 numerical inputs')
        if b['y'] is not None:b['target']=np.asarray([CLASSES.index(y) for y in b['y']],dtype=np.int64)
    versions={k:importlib.metadata.version(k) for k in ['numpy','scikit-learn','lightgbm','catboost']}
    if versions!=proposal['versions']:raise RuntimeError(f'Expected pinned versions {proposal["versions"]}, got {versions}')
    identity=hashlib.sha256((json.dumps(manifest,sort_keys=True)+json.dumps(versions,sort_keys=True)).encode()).hexdigest()
    return panel,proposal,identity,versions


def baseline(c):
    p17=np.load(HERE/'probs'/f'{c}.npz',allow_pickle=False)['proba']
    p4=np.load(HERE/'rank4'/f'{c}.npz',allow_pickle=False)['proba']
    vector=np.asarray(json.loads((HERE/'pool_r4full_vector.json').read_text())['vR'])
    p=combine(p17,p4)*vector;return p/p.sum(1,keepdims=True)

def metric(b,p):return float(f1_score(b['y'],np.asarray(CLASSES)[p.argmax(1)],average='weighted',zero_division=0))


def stack(panel,c):
    latest=shift(c,-3);limit=shift(latest,-3)
    origins=[o for o in panel['monthly_cutoffs'] if o<=latest]
    x=np.vstack([panel['cutoffs'][o]['X'] for o in origins]);y=np.concatenate([panel['cutoffs'][o]['target'] for o in origins])
    ids=np.concatenate([panel['cutoffs'][o]['ids'] for o in origins]);w=[];dates=[]
    for o in origins:
        yy,mm=map(int,o[:7].split('-'));w.extend([.70+.05*((yy-2024)*12+mm-9)]*len(panel['cutoffs'][o]['ids']));dates.extend([o]*len(panel['cutoffs'][o]['ids']))
    dates=np.asarray(dates);tr=dates<=limit;va=dates==latest
    if not tr.any() or not va.any() or not all(shift(o,3)<=latest for o in origins if o<=limit):raise ValueError('invalid purged split')
    return x,y,ids,np.asarray(w),tr,va,origins,dates


def fit_cat(x,y,ids,w,tr,va,predict_x,predict_ids,params):
    from catboost import CatBoostClassifier,FeaturesData,Pool
    def pool(a,b,yy=None,ww=None):
        return Pool(FeaturesData(num_feature_data=np.ascontiguousarray(a,dtype=np.float32),cat_feature_data=np.asarray(b,dtype=object).reshape(-1,1)),label=yy,weight=ww)
    inner=CatBoostClassifier(**params)
    inner.fit(pool(x[tr],ids[tr],y[tr],w[tr]),eval_set=pool(x[va],ids[va],y[va]),early_stopping_rounds=80,use_best_model=True,verbose=100)
    rounds=max(int(inner.best_iteration_)+1,80)
    full=CatBoostClassifier(**dict(params,iterations=rounds))
    full.fit(pool(x,ids,y,w),verbose=100)
    raw=full.predict_proba(pool(predict_x,predict_ids));out=np.zeros((len(predict_ids),17))
    for j,c in enumerate(full.classes_):out[:,int(c)]=raw[:,j]
    return valid(out,len(out)),{'rounds':rounds,'classes':list(map(int,full.classes_))},full


def fit_rank(panel,c,origins,y,w,dates,params):
    from lightgbm import LGBMRanker,early_stopping
    x=np.vstack([panel['cutoffs'][o]['rank_x'] for o in origins]);candidate=np.tile(np.arange(17),len(y))
    target=(candidate==np.repeat(y,17)).astype(np.int32);weights=np.repeat(w,17)
    latest=shift(c,-3);limit=shift(latest,-3);tr=np.repeat(dates<=limit,17);va=np.repeat(dates==latest,17)
    cat=[panel['cutoffs'][c]['rank_label_column']]
    model=LGBMRanker(**params)
    model.fit(x[tr],target[tr],group=np.full(int(tr.sum()/17),17),sample_weight=weights[tr],
        eval_set=[(x[va],target[va])],eval_group=[np.full(int(va.sum()/17),17)],eval_at=[1],
        categorical_feature=cat,callbacks=[early_stopping(60,verbose=False)])
    rounds=max(int(model.best_iteration_ or 80),80)
    full=LGBMRanker(**dict(params,n_estimators=rounds))
    full.fit(x,target,group=np.full(len(y),17),sample_weight=weights,categorical_feature=cat)
    logits=full.booster_.predict(panel['cutoffs'][c]['rank_x']).reshape(-1,17)
    logits-=logits.max(1,keepdims=True);p=np.exp(logits);p/=p.sum(1,keepdims=True)
    return valid(p,len(p)),{'rounds':rounds,'candidate_rows':len(x),'choice_features':x.shape[1]},full


def path_for(arm,c):return HERE/'predictions'/arm/f'{c}.npz'

def child_fit(panel,proposal,identity,arm,c):
    started=time.monotonic();x,y,ids,w,tr,va,origins,dates=stack(panel,c)
    if arm=='rank17':p,meta,model=fit_rank(panel,c,origins,y,w,dates,proposal['arms'][arm]['params'])
    else:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count()<1:raise RuntimeError('GPU required for CatBoost arms; select T4 runtime')
        p,meta,model=fit_cat(x,y,ids,w,tr,va,panel['cutoffs'][c]['X'],panel['cutoffs'][c]['ids'],proposal['arms'][arm]['params'])
    dest=path_for(arm,c);dest.parent.mkdir(parents=True,exist_ok=True)
    model_path=dest.with_suffix('.txt' if arm=='rank17' else '.cbm')
    if arm=='rank17':model.booster_.save_model(str(model_path))
    else:model.save_model(str(model_path))
    meta.update({'origin':c,'origins':origins,'full_training_rows':len(y),'seconds':round(time.monotonic()-started,1),'model_sha256':sha(model_path)})
    tmp=dest.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,proba=p,ids=panel['cutoffs'][c]['ids'].astype(str),identity=identity,metadata=json.dumps(meta))
    tmp.replace(dest);log(f'FIT DONE {arm} {c}: {meta}')


def prediction(panel,proposal,identity,arm,c,deadline):
    dest=path_for(arm,c)
    if not dest.exists():
        remaining=deadline-time.monotonic()
        if remaining<1:raise TimeoutError('global run budget exhausted')
        subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--fit',arm,c],check=True,
            timeout=min(proposal['per_origin_seconds'],remaining))
    with np.load(dest,allow_pickle=False) as d:
        if str(d['identity'])!=identity or not np.array_equal(d['ids'],panel['cutoffs'][c]['ids']):raise RuntimeError('incompatible prediction cache '+str(dest))
        p=valid(d['proba'],len(d['ids']));meta=json.loads(str(d['metadata']))
    model_path=dest.with_suffix('.txt' if arm=='rank17' else '.cbm')
    if sha(model_path)!=meta['model_sha256']:raise RuntimeError('saved model hash mismatch')
    return p,meta


def export(panel,candidate,tag,decision,exploratory):
    with (HERE/'sasol_sprint16_r4full_vR.csv').open(newline='') as f:source=list(csv.reader(f))
    header,rows=source[0],source[1:];ids=panel['cutoffs'][TEST]['ids'].astype(str).tolist()
    if len(rows)!=5488 or len(set(ids))!=5488 or [r[0] for r in rows]!=ids:raise ValueError('official IDs/order mismatch')
    label=np.asarray(CLASSES)[candidate.argmax(1)];new=[r[:3]+[str(v)] for r,v in zip(rows,label)]
    if any(a[:3]!=b[:3] for a,b in zip(rows,new)):raise ValueError('CLV changed')
    values=np.asarray([[float(r[1]),float(r[2])] for r in new])
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('invalid CLV')
    dest=HERE/('exploratory' if exploratory else 'validated')/(('exploratory_' if exploratory else 'validated_')+tag+'.csv');dest.parent.mkdir(exist_ok=True)
    with dest.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(new)
    decision['exports'].append({'file':str(dest.relative_to(HERE)),'exploratory_only':exploratory,
        'changes':int(sum(r[3]!=v for r,v in zip(rows,label))),'sha256':sha(dest),
        'source_sha256':sha(HERE/'sasol_sprint16_r4full_vR.csv')})


def summarize(gains):return {'gains':gains,'mean_gain':float(np.mean(list(gains.values()))),'min_gain':min(gains.values())}
def passes(s,gate):return s['mean_gain']>=gate['mean_gain_min'] and s['min_gain']>=gate['per_origin_min']
def probe_pass(s,rule):return s['mean_gain']>=rule['mean_gain_min'] and s['min_gain']>=rule['per_origin_min'] and all(s['gains'][c]>0 for c in rule['positive_origins'])


def run(panel,proposal,identity,versions):
    out=HERE/'moonshot_decision.json';deadline=time.monotonic()+proposal['global_seconds']
    decision={'proposal_sha256':sha(HERE/'proposal.json'),'identity':identity,'versions':versions,'arms':{},'exports':[],
        'baseline':{},'note':'Six predeclared views, three model families. Exploratory exports are not promoted models or validated public gains.'}
    for c in POOL+EVAL:
        score=metric(panel['cutoffs'][c],baseline(c));decision['baseline'][c]=score
        if abs(score-BASELINE_GUARD[c])>1e-9:raise ValueError('baseline mismatch '+c)
    save(out,decision)
    for arm in ARMS:
        d={'screen':{},'folds':{},'errors':[]};decision['arms'][arm]=d
        names=[arm,arm+'_half'];screen={name:{} for name in names}
        def view(p,c,name):return p if name==arm else .5*p+.5*baseline(c)
        try:
            for c in POOL:
                p,meta=prediction(panel,proposal,identity,arm,c,deadline)
                for name in names:
                    score=metric(panel['cutoffs'][c],view(p,c,name));screen[name][c]=score-decision['baseline'][c]
                    log(f'{name} {c}: F1 {score:.6f}, gain {screen[name][c]:+.6f}')
                d.setdefault('fits',{})[c]=meta;d['screen_partial']=screen;save(out,decision)
            for name in names:
                s=summarize(screen[name]);s['passed']=passes(s,proposal['screen_gate']);s['probe_eligible']=probe_pass(s,proposal['probe_gate']);d['screen'][name]=s
            strict=[name for name in names if d['screen'][name]['passed']]
            if strict:
                folds={name:{} for name in strict}
                for c in EVAL:
                    p,meta=prediction(panel,proposal,identity,arm,c,deadline);d['fits'][c]=meta
                    for name in strict:folds[name][c]=metric(panel['cutoffs'][c],view(p,c,name))-decision['baseline'][c]
                for name in strict:
                    s=summarize(folds[name]);s['passed']=passes(s,proposal['eval_gate']);d['folds'][name]=s
            promoted=[name for name in names if d['folds'].get(name,{}).get('passed',False)]
            exploratory=[name for name in names if name not in promoted and d['screen'][name]['probe_eligible']]
            if promoted or exploratory:
                p,meta=prediction(panel,proposal,identity,arm,TEST,deadline);d['fits'][TEST]=meta
                for name in promoted+exploratory:export(panel,view(p,TEST,name),name,decision,name not in promoted)
        except Exception as e:
            d['errors'].append(type(e).__name__+': '+str(e));log(f'ARM STOPPED {arm}: {d["errors"][-1]}')
        save(out,decision);gc.collect()
        if time.monotonic()>=deadline:break
    print('\n===== COPY THIS BACK =====');print(json.dumps(decision,indent=2));return decision


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');parser.add_argument('--fit',nargs=2);args=parser.parse_args()
    panel,proposal,identity,versions=inputs()
    if args.fit:
        arm,c=args.fit
        if arm not in ARMS or c not in POOL+EVAL+[TEST]:raise ValueError('unregistered fit')
        child_fit(panel,proposal,identity,arm,c);return
    if args.check:
        for c in POOL+EVAL:
            value=metric(panel['cutoffs'][c],baseline(c));*_,origins,dates=stack(panel,c)
            if abs(value-BASELINE_GUARD[c])>1e-9:raise ValueError('baseline mismatch')
            log(f'CHECK {c}: baseline {value:.6f}, 794 numeric features + customer ID, 17-choice ranker, purged split OK')
        log('PASS: frozen hashes, all five blocks, original baseline scores and training splits; no fitting');return
    run(panel,proposal,identity,versions)


if __name__=='__main__':main()
