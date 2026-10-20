"""Runner verification: synthetic fits and gates; no real-customer candidate fits."""
import argparse
import copy
import shutil
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bundle-dir',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    stage=Path(__file__).resolve().parent
    sys.path.insert(0,str(args.bundle_dir))
    import colab_shop_cadence as runner
    panel,proposal,vector,manifest=runner.read_inputs()
    decision={}
    baseline4=runner.base4(runner.TEST,len(panel['cutoffs'][runner.TEST]['ids']))
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);(folder/'probs').mkdir()
        shutil.copy2(args.bundle_dir/'sasol_sprint16_r4full_vR.csv',folder/'sasol_sprint16_r4full_vR.csv')
        shutil.copy2(args.bundle_dir/'probs'/f'{runner.TEST}.npz',folder/'probs'/f'{runner.TEST}.npz')
        with patch.object(runner,'HERE',folder):runner.export(panel,baseline4,vector,decision)
    assert decision['exported']['rows']==5488 and decision['exported']['changes_vs_best']==0
    # Hand-controlled score outcomes exercise routing independently of classifiers.
    baseline={c:0.4 for c in runner.POOL+runner.EVAL}
    scenarios=[(0.0,0.0,runner.POOL,False),(0.005,0.0,runner.POOL+runner.EVAL,False),
               (0.005,0.005,runner.POOL+runner.EVAL+[runner.TEST],True)]
    for screen_gain,fold_gain,expected,exported in scenarios:
        calls=[];exports=[]
        def fit(*args):
            calls.append(args[1]);return np.full((1,4),.25),{'original_fit_seconds':0.0}
        def score(_panel,cutoff,*args):return .4+(screen_gain if cutoff in runner.POOL else fold_gain)
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'fit_origin',side_effect=fit),patch.object(runner,'score',side_effect=score),patch.object(runner,'export',side_effect=lambda *a:exports.append(True)):
            d={'members':{}}
            runner.run_stages(panel,proposal,vector,baseline,d,Path(tmp)/'decision.json','synthetic-gate-test')
            assert calls==expected and bool(exports)==exported
    # Actual LightGBM path on generated random rows only, including cache reuse.
    rng=np.random.default_rng(7)
    toy={'monthly_cutoffs':['2024-09-01','2024-10-01','2024-11-01','2024-12-01'],'cutoffs':{}}
    labels=['Inactivity','Stable','Existing-category growth: Fuel','New category adoption: Beverages']*10
    for c in toy['monthly_cutoffs']+['2025-03-01']:
        toy['cutoffs'][c]={'X_extended':rng.normal(size=(40,326)),'y':labels,
                           'ids':np.asarray([f'{i:03}' for i in range(40)])}
    spec=copy.deepcopy(proposal);spec['head'].update(n_estimators=3,min_child_samples=2,minimum_refit_rounds=2,early_stopping_rounds=2,seeds=[7])
    with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'HERE',Path(tmp)):
        pred,meta=runner.fit_origin(toy,'2025-03-01',spec,'synthetic-fit-test')
        assert pred.shape==(40,4) and np.allclose(pred.sum(1),1)
        cached,cm=runner.fit_origin(toy,'2025-03-01',spec,'synthetic-fit-test')
        np.testing.assert_array_equal(pred,cached);assert cm==meta
        try:runner.fit_origin(toy,'2025-03-01',spec,'wrong-identity')
        except RuntimeError:pass
        else:raise AssertionError('incompatible prediction cache accepted')
    result={'passed':True,'baseline_guards':7,'original_precision':'float64',
        'export_smoke_rows':5488,'export_smoke_changes':0,'failed_screen_skips_eval':True,
        'failed_eval_skips_december':True,'all_pass_runs_december':True,'synthetic_lightgbm_fit_and_cache':True,
        'incompatible_cache_rejected':True,'real_customer_candidate_fits':0}
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
