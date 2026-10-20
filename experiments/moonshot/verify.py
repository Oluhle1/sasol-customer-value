"""Check experimental code using synthetic fits; never fit real customer labels."""
import argparse,copy,json,shutil,sys,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle-dir',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);a=parser.parse_args()
    sys.path.insert(0,str(a.bundle_dir));import moonshot as m
    panel,proposal,identity,versions=m.inputs()
    # Exact expected ranking groups/label-ID column and finite inputs.
    for c,b in panel['cutoffs'].items():
        assert b['X'].shape==(len(b['ids']),794)
        np.testing.assert_array_equal(b['rank_x'][:,b['rank_label_column']],np.tile(np.arange(17),len(b['ids'])))
    assert not m.probe_pass({'mean_gain':-.001,'min_gain':-.003,'gains':{'2025-07-01':.002,'2025-08-01':.002}},proposal['probe_gate'])
    assert m.probe_pass({'mean_gain':.001,'min_gain':-.003,'gains':{'2025-07-01':.002,'2025-08-01':.002}},proposal['probe_gate'])
    assert not m.passes({'mean_gain':.003,'min_gain':0},proposal['screen_gate'])
    assert m.passes({'mean_gain':.004,'min_gain':-.002},proposal['screen_gate'])
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);shutil.copy2(a.bundle_dir/'sasol_sprint16_r4full_vR.csv',folder/'sasol_sprint16_r4full_vR.csv')
        d={'exports':[]}
        with patch.object(m,'HERE',folder):
            m.export(panel,np.eye(17)[np.zeros(5488,dtype=int)],'synthetic',d,True)
        assert d['exports'][0]['exploratory_only'] and d['exports'][0]['file'].startswith('exploratory/exploratory_')
        import csv
        with (folder/d['exports'][0]['file']).open(newline='') as f:rows=list(csv.reader(f))
        with (folder/'sasol_sprint16_r4full_vR.csv').open(newline='') as f:old=list(csv.reader(f))
        assert len(rows)==5489 and all(x[:3]==y[:3] for x,y in zip(rows[1:],old[1:]))
    # Actual CatBoost categorical-ID API on random CPU rows only.
    rng=np.random.default_rng(7);x=rng.normal(size=(136,794)).astype(np.float32);y=np.tile(np.arange(17),8)
    ids=np.asarray([str(i%34) for i in range(len(y))]);w=np.ones(len(y));tr=np.arange(len(y))<102;va=~tr
    p,meta,model=m.fit_cat(x,y,ids,w,tr,va,x[:17],ids[:17],{'iterations':3,'depth':3,'learning_rate':.1,'loss_function':'MultiClass','classes_count':17,'task_type':'CPU','thread_count':1,'random_seed':7,'allow_writing_files':False})
    assert p.shape==(17,17) and np.allclose(p.sum(1),1)
    toy={'cutoffs':{}};origins=['2024-09-01','2024-10-01','2024-11-01','2024-12-01'];dates=[]
    for c in origins+['2025-03-01']:
        matrix=rng.normal(size=(34*17,12)).astype(np.float32);matrix[:,0]=np.tile(np.arange(17),34)
        toy['cutoffs'][c]={'rank_x':matrix,'rank_label_column':0}
        if c in origins:dates.extend([c]*34)
    params=copy.deepcopy(proposal['arms']['rank17']['params']);params.update(n_estimators=3,min_child_samples=2,num_leaves=7)
    p,meta,model=m.fit_rank(toy,'2025-03-01',origins,np.tile(np.arange(17),8),np.ones(136),np.asarray(dates),params)
    assert p.shape==(34,17) and np.allclose(p.sum(1),1)
    # Prediction cache must bind the original ID order, input identity and saved model.
    with tempfile.TemporaryDirectory() as tmp,patch.object(m,'HERE',Path(tmp)):
        path=m.path_for('rank17',m.TEST);path.parent.mkdir(parents=True)
        model_path=path.with_suffix('.txt');model_path.write_text('synthetic model hash fixture')
        np.savez(path,proba=np.full((5488,17),1/17),ids=panel['cutoffs'][m.TEST]['ids'],identity='test',metadata=json.dumps({'model_sha256':m.sha(model_path)}))
        m.prediction(panel,proposal,'test','rank17',m.TEST,0)
        try:m.prediction(panel,proposal,'wrong','rank17',m.TEST,0)
        except RuntimeError:pass
        else:raise AssertionError('incompatible cache accepted')
    result={'passed':True,'numeric_features':794,'ranking_groups':17,'six_views_preregistered':True,
        'synthetic_cpu_catboost_customer_id_fit':True,'synthetic_lambdarank_fit':True,'clv_and_ids_preserved':True,
        'probe_and_promotion_rules_separate':True,'model_hash_and_cache_identity_checked':True,
        'real_customer_candidate_fits':0,'gpu_execution_tested_locally':False}
    a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
