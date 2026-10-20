"""Cross-check constructed arrays against the immutable panel and raw audit counts."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
from load_blocks import load_blocks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    out=args.out or args.root/'outputs/feature_rebuild'
    report=json.loads((out/'feature_build_report.json').read_text())
    with (args.root/'outputs/oct10/category_dynamics_panel.pkl').open('rb') as f:panel=pickle.load(f)
    compared=0
    for cutoff,frame in panel['frames'].items():
        extra,names=load_blocks(out,cutoff,frame.index,list(report['features_per_block']))
        assert extra.shape==(len(frame),489) and len(set(names))==489
        assert not set(names)&set(panel['feature_cols'])
        positions={c:i for i,c in enumerate(names)}
        for col in frame:
            if col.startswith('cat_') and col.endswith('_q1'):
                cat=col[4:-3]
                expected=np.log1p(np.maximum(extra[:,positions[f'cs_{cat}_q1_net_rand']],0))
                np.testing.assert_allclose(expected,frame[col],atol=1e-6,rtol=1e-6)
                compared+=1
        np.testing.assert_array_equal(extra[:,positions['cs_categories_ever']],frame.n_cat_ever)
        # Consumers must fail on wrong order, rather than silently mixing customers.
        try:load_blocks(out,cutoff,frame.index[::-1],['shop_cadence'])
        except RuntimeError:pass
        else:raise AssertionError('consumer accepted reversed IDs')
    ids=panel['frames']['2025-12-01'].index
    x,names=load_blocks(out,'2025-12-01',ids,['basket_attachment'])
    def col(name):return x[:,names.index(name)]
    both=(col('ba_fuel_n_q1')>0)&(col('ba_shop_n_q1')>0)
    assert int(col('ba_mixed_n_q1').sum())==1944
    assert int(both.sum())==1835
    assert int((both&(col('ba_mixed_n_q1')==0)).sum())==970
    evidence={'passed':True,'cutoffs':len(panel['frames']),'arrays_checked':len(panel['frames'])*5,
              'new_features':489,'category_spend_reconciliations':compared,
              'category_breadth_matches_baseline':True,'reversed_ids_rejected':True,
              'december_attachment_matches_independent_raw_audit':True,
              'candidate_models_fitted':0}
    (out/'consumer_verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
