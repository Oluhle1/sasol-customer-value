"""Small semantic edge cases for the line/basket/customer feature pipeline."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from build_layers import enrich_lines, build_baskets, build_customer_blocks, BLOCKS


def run(root):
    sys.path.insert(0,str(root/'src'))
    from reference import label_rules as lr
    cfg=json.loads((root/'configs/label_config.json').read_text())
    rows=[]
    def line(customer,receipt,day,cat,amount,qty='1',kind='Sale',name='product',site='s1',sequence=None,total='100'):
        rows.append({'DateTimeZA':day,'ID':customer,'TransactionId':receipt,
            'LineItemSequence':str(sequence or (sum(r['TransactionId']==receipt and r['ID']==customer for r in rows)+1)),
            'LocationId':site,'LineItemType':kind,'TransactionTotalInclAmount':total,
            'StockItemId':cat+'sku','ItemName':name,'ItemCategoryLevel1':'test',
            'ItemCategoryLevel2':cat,'ItemCategoryLevel3':cat+'fine','Quantity':qty,
            'TotalExclAmount':str(amount),'TotalModifierAmount':'0','UnitSellingExclPrice':str(amount)})
    # Shared receipt and category: counts must be baskets, not lines.
    line('a','mixed','2025-01-10','Fuel','50',kind='Sale_Fuel')
    line('a','mixed','2025-01-10','Beverages','20')
    line('a','mixed','2025-01-10','Beverages','10')
    # A return decreases net shop amount but is not a qualifying purchase.
    line('a','return','2025-02-10','Beverages','-5',qty='-1')
    line('a','shop','2024-12-01','Beverages','10')
    line('a','old','2024-08-01','Beverages','5')
    line('a','fuelonly','2025-01-11','Fuel','70',kind='Sale_Fuel')
    # Literal NA/Unknown are included categories. Print media is eligibility only.
    line('b','na','2025-01-01','NA','3')
    line('b','unknown','2025-01-02','Unknown','4')
    line('b','excluded','2025-01-03','Print media','8')
    line('b','adjustment','2025-01-04','Beverages','9',name=cfg['adjustment_items'][0])
    # Return-only category and zero-price line do not create purchase history.
    line('b','returnonly','2025-01-05','Snacking','-2',qty='-1')
    line('b','free','2025-01-06','Food & beverage','0')
    # First recent category exposure must not be called reactivation.
    line('c','first','2025-01-07','Tobacco','11')
    # Same-day fuel and shop on separate receipts is not a mixed basket.
    line('d','fuel','2025-01-08','Fuel','40',kind='Sale_Fuel')
    line('d','shop','2025-01-08','Snacking','7')
    # Two zero-time-gap shop receipts need an availability flag, never infinity.
    line('e','one','2025-01-09','Beverages','2')
    line('e','two','2025-01-09','Beverages','3')
    # Different Other-mapped source categories must stay distinct.
    line('f','known','2024-08-01','Car Wash','10')
    line('f','adopt','2025-01-10','Krispy Kreme','12')
    # Exact cutoff and future lines are excluded before aggregation.
    line('a','boundary','2025-03-01','Groceries','900')
    line('a','future','2025-03-10','Fuel','10000',kind='Sale_Fuel')
    raw=lr.prepare(pd.DataFrame(rows))
    d=enrich_lines(raw,cfg)
    b,c,checks=build_baskets(d)
    assert all(checks.values())
    receipt=b[(b.ID=='a')&(b.TransactionId=='mixed')].iloc[0]
    assert receipt.source_total_incl==100 and receipt.line_count==3 and receipt.shop_scaled==30*lr.SCALE
    assert receipt.purchase_categories==2 and receipt.mixed
    assert not b[b.TransactionId=='excluded'].qualifying_purchase.any()
    assert not b[b.TransactionId=='adjustment'].qualifying_purchase.any()
    ids=pd.Index(['a','b','c','d','e','f','no_history'],name='ID')
    cutoff='2025-03-01';start=pd.Timestamp('2024-03-19')
    blocks=build_customer_blocks(d,b,c,cutoff,ids,cfg,start)
    frames={name:value.frame() for name,value in blocks.items()}
    category=frames['category_state'];attach=frames['basket_attachment'];shop=frames['shop_cadence']
    assert category.loc['a','cs_beverages_q1_purchase_baskets']==2
    assert category.loc['a','cs_beverages_q1_net_rand']==35
    assert category.loc['a','cs_groceries_q1_purchase_baskets']==0
    assert category.loc['b','cs_na_ever']==1 and category.loc['b','cs_unknown_ever']==1
    assert category.loc['b','cs_snacking_ever']==0 and category.loc['b','cs_food_and_beverage_ever']==0
    assert category.loc['c','cs_tobacco_first_observed_recent']==1
    assert category.loc['c','cs_tobacco_reactivated']==0
    assert category.loc['f','cs_krispy_kreme_first_observed_recent']==1
    assert category.loc['f','cs_car_wash_ever']==1
    assert category.loc['a','cs_categories_ever']+category.loc['a','cs_categories_unseen']==24
    assert attach.loc['d','ba_mixed_n_q1']==0 and attach.loc['d','ba_same_day_both_n_q1']==1
    assert shop.loc['e','sc_corrected_cadence_unavailable']==1
    assert frames['coverage_and_ratios'].loc['no_history','cr_q1_customer_observed_exposure']==0
    for name,frame in frames.items():
        assert frame.index.equals(ids) and np.isfinite(frame.to_numpy()).all()
    truncated=enrich_lines(raw[raw._time<pd.Timestamp(cutoff)].copy(),cfg)
    bt,ct,_=build_baskets(truncated)
    again=build_customer_blocks(truncated,bt,ct,cutoff,ids,cfg,start)
    for name in BLOCKS:
        pd.testing.assert_frame_equal(frames[name],again[name].frame(),check_exact=True)
    # A receipt spanning the cutoff must fail rather than pull in a future line.
    broken=raw.copy()
    idx=broken.index[broken.TransactionId=='mixed'][0]
    broken.loc[idx,'_time']=pd.Timestamp('2025-03-01')
    try:
        build_baskets(enrich_lines(broken,cfg))
    except ValueError as exc:
        assert 'inconsistent' in str(exc)
    else:
        raise AssertionError('inconsistent basket time accepted')
    print('PASS semantic edge cases: signed returns, basket counts/totals, eligibility versus purchase scope, literal NA/Unknown, missing gaps, category states, original Other categories, exact cutoff/future deletion, ID order and inconsistent receipt rejection')
    return {'passed':True,'synthetic_lines':len(raw),'blocks':{name:len(x.values) for name,x in blocks.items()}}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args();print(json.dumps(run(args.root),indent=2))
