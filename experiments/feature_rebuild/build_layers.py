"""Past-only feature layers built from released lines, receipts and customer histories.

No model fits, label generation, baseline mutation, or submission exports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = 'feature_layers_v1'
BLOCKS = ['shop_cadence', 'basket_attachment', 'category_state',
          'coverage_and_ratios', 'fine_category_and_site']
SCALE = 1_000_000
KEY = ['ID', 'TransactionId']


def log(message):
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_pickle(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        pickle.dump(value, f, protocol=4)
    temp.replace(path)


def construct_spec(config):
    return {
        'version': VERSION,
        'scope': 'Feature construction and integrity checks only; no model fitting.',
        'layers': ['typed_original_lines', 'baskets_and_original_category_baskets',
                   'customer_cutoff_blocks'],
        'blocks': BLOCKS,
        'shop_cadence_definition': 'Any non-adjustment in-scope nonfuel receipt, matching the original shop cadence event definition; qualifying positive purchase counts are separate.',
        'attachment_definition': 'Same-receipt positive fuel Sale_Fuel and positive in-scope nonfuel purchase flags. Same-day association is separate, not assumed to be a visit.',
        'windows': 'Lifetime, 30 days, preceding three calendar months q1, preceding calendar quarter q2, six calendar months as specified by each block.',
        'category_definition': 'Original included Level2 categories, positive quantity AND amount for purchase flags; signed unmodified source amounts for net spend. No Other-pooling.',
        'missing_values': 'Finite stored values with explicit availability/denominator flags; unseen recency = 999 days, with ever/never indicators.',
        'scaling': 'No fitted scalers. Log1p positive ratios and signed-log monetary changes only; downstream scaling must be trained on eligible training origins.',
        'growth_features': 'Separate known absolute-spend requirement and relative-spend boundary from config; not future labels or approximated label decisions.',
        'minimum_historical_transactions': config['minimum_historical_transactions'],
        'future_opportunity_acceptance_gate': {
            'screen_origins': ['2025-03-01', '2025-04-01', '2025-05-01', '2025-07-01', '2025-08-01'],
            'screen_mean_gain_min': .004, 'screen_per_origin_min': -.002,
            'eval_origins': ['2025-06-01', '2025-09-01'],
            'eval_mean_gain_min': .004, 'eval_per_origin_min': .0015,
            'protocol': 'One block at a time, unchanged model/rows/seeds/blend/vR, matched baseline. Preregister a separate fitting proposal before training. No June/September-based feature selection.'},
        'clv': 'No CLV experiment authorized by this construction specification; requires a separate preregistered regression comparison.'}


def enrich_lines(raw, config):
    """raw must be prepared by reference.label_rules.prepare/load_transactions."""
    d = raw.copy()
    if d.duplicated(KEY + ['LineItemSequence']).any():
        raise ValueError('duplicate customer/receipt/line keys')
    included = [c for c, v in config['category_mapping'].items() if v['included']]
    unknown = set(d.ItemCategoryLevel2.astype(str)) - set(config['category_mapping'])
    if unknown:
        raise ValueError(f'unknown original categories: {unknown}')
    d['_adjustment'] = d.ItemName.isin(config['adjustment_items'])
    d['_in_scope'] = ~d._adjustment & d.ItemCategoryLevel2.isin(included)
    d['_purchase'] = d._in_scope & d._quantity.gt(0) & d._amount.gt(0)
    d['_fuel_line'] = ~d._adjustment & d.ItemCategoryLevel2.eq('Fuel') & d.LineItemType.eq('Sale_Fuel')
    d['_shop_line'] = d._in_scope & ~d.ItemCategoryLevel2.eq('Fuel')
    d['_fuel_purchase'] = d._purchase & d._fuel_line
    d['_shop_purchase'] = d._purchase & d._shop_line
    d['_fuel_scaled'] = d._quantity.where(d._fuel_line, 0)
    d['_shop_scaled'] = d._amount.where(d._shop_line, 0)
    d['_category_scaled'] = d._amount.where(d._in_scope, 0)
    d['_source_basket_total'] = pd.to_numeric(d.TransactionTotalInclAmount, errors='raise')
    if not np.isfinite(d._source_basket_total).all():
        raise ValueError('nonfinite source basket amount')
    # Only low-cardinality source strings are categorical; customer/receipt IDs stay strings.
    for c in ['LocationId', 'LineItemType', 'StockItemId', 'ItemName',
              'ItemCategoryLevel1', 'ItemCategoryLevel2', 'ItemCategoryLevel3']:
        d[c] = d[c].astype('category')
    return d


def build_baskets(lines):
    """All baskets count for eligibility; purchase flags keep target scope separate."""
    g = lines.groupby(KEY, sort=False, observed=True)
    consistency = g[['_time', 'LocationId', 'TransactionTotalInclAmount']].nunique(dropna=False)
    if (consistency > 1).any().any():
        raise ValueError('receipt timestamp/site/total is inconsistent; refuse aggregation across cutoff')
    b = g.agg(time=('_time', 'first'), site=('LocationId', 'first'),
              source_total_incl=('_source_basket_total', 'first'),
              line_count=('LineItemSequence', 'size'),
              fuel_any=('_fuel_line', 'max'), shop_any=('_shop_line', 'max'),
              fuel_purchase=('_fuel_purchase', 'max'), shop_purchase=('_shop_purchase', 'max'),
              qualifying_purchase=('_purchase', 'max'),
              fuel_scaled=('_fuel_scaled', 'sum'), shop_scaled=('_shop_scaled', 'sum'))
    positive = lines[lines._purchase]
    breadth = positive.groupby(KEY, sort=False, observed=True).agg(
        purchase_categories=('ItemCategoryLevel2', 'nunique'),
        purchase_products=('StockItemId', 'nunique'))
    b = b.join(breadth).fillna({'purchase_categories': 0, 'purchase_products': 0}).reset_index()
    b['site'] = b.site.astype(str)
    b['mixed'] = b.fuel_purchase & b.shop_purchase
    b['day'] = b.time.dt.floor('D')
    c = lines[lines._in_scope].groupby(KEY + ['ItemCategoryLevel2'], sort=False, observed=True).agg(
        time=('_time', 'first'), net_scaled=('_amount', 'sum'), purchase=('_purchase', 'max')).reset_index()
    c = c.rename(columns={'ItemCategoryLevel2': 'category'})
    c['category'] = c.category.astype(str)
    c['day'] = c.time.dt.floor('D')
    checks = {
        'unique_baskets': not b.duplicated(KEY).any(),
        'unique_category_baskets': not c.duplicated(KEY + ['category']).any(),
        'fuel_scaled_reconciles': int(b.fuel_scaled.sum()) == int(lines._fuel_scaled.sum()),
        'nonfuel_scaled_reconciles': int(b.shop_scaled.sum()) == int(lines._shop_scaled.sum()),
        'category_scaled_reconciles': int(c.net_scaled.sum()) == int(lines._category_scaled.sum()),
        'basket_total_counted_once': float(b.source_total_incl.sum()) == float(g._source_basket_total.first().sum()),
        'receipt_timestamp_site_total_consistent': True,
        'purchase_count_is_distinct_baskets': True}
    if not all(checks.values()):
        raise ValueError(f'failed basket checks: {checks}')
    return b, c, checks


class Features:
    def __init__(self, ids):
        self.ids = pd.Index(ids, dtype=str, name='ID')
        self.values = {}
        self.catalog = {}

    def aligned(self, values, fill=0):
        if isinstance(values, pd.Series):
            return values.reindex(self.ids).fillna(fill).to_numpy(dtype=float)
        if np.isscalar(values):
            return np.full(len(self.ids), values, dtype=float)
        values = np.asarray(values, dtype=float)
        if len(values) != len(self.ids):
            raise ValueError('feature row count mismatch')
        return np.where(np.isfinite(values), values, fill)

    def add(self, name, values, meaning, unit='count', fill=0):
        if name in self.values:
            raise ValueError(f'duplicate feature {name}')
        self.values[name] = self.aligned(values, fill)
        self.catalog[name] = {'meaning': meaning, 'unit': unit, 'missing_fill': fill}
        return self.values[name]

    def ratio(self, name, numerator, denominator, meaning):
        num, den = self.aligned(numerator), self.aligned(denominator)
        valid = den > 0
        result = np.divide(num, den, out=np.zeros_like(num), where=valid)
        self.add(name, result, meaning, 'ratio')
        self.add(name + '_undefined', ~valid, 'Denominator is not positive', 'flag')
        return result

    def frame(self):
        f = pd.DataFrame(self.values, index=self.ids)
        if not np.isfinite(f.to_numpy()).all():
            raise ValueError('nonfinite generated feature')
        return f


def counts(rows):
    return rows.groupby('ID', observed=True).size()


def gap_summary(rows):
    s = rows.sort_values(['ID', 'time', 'TransactionId']).copy()
    s['_gap'] = s.groupby('ID', observed=True).time.diff().dt.total_seconds() / 86400
    g = s.groupby('ID', observed=True)._gap
    return pd.DataFrame({'mean': g.mean(), 'median': g.median(), 'std': g.std(ddof=0),
                         'p90': g.quantile(.9), 'n_gaps': g.count()})


def recency(rows, t):
    return (t - rows.groupby('ID', observed=True).time.max()).dt.total_seconds() / 86400


def in_window(rows, start, end):
    return rows[(rows.time >= start) & (rows.time < end)]


def shop_cadence(b, c, lines, t, ids, config, data_start):
    f = Features(ids)
    shop = b[b.shop_any]
    f.add('sc_any_receipts_life', counts(shop), 'Any non-adjustment shop receipt')
    f.add('sc_qualifying_receipts_life', counts(b[b.shop_purchase]), 'Positive qualifying shop receipt')
    rec = f.add('sc_recency_days', recency(shop, t), 'Days since any shop receipt', 'days', fill=999)
    f.add('sc_never_shop', ~f.ids.isin(shop.ID), 'No historical shop receipt', 'flag')
    lifetime_gap = None
    lifetime_available = None
    for suffix, events in [('life', shop), ('6m', shop[shop.time >= t - pd.DateOffset(months=6)])]:
        stats = gap_summary(events)
        for metric in ['mean', 'median', 'std', 'p90']:
            f.add(f'sc_gap_{metric}_{suffix}', stats[metric], f'Shop-only {suffix} inter-receipt gap {metric}', 'days')
        n_gap = f.add(f'sc_n_gaps_{suffix}', stats.n_gaps, f'Observed shop {suffix} gaps')
        mean = f.aligned(stats['mean'])
        available = (n_gap > 0) & (mean > 0)
        f.add(f'sc_gap_unavailable_{suffix}', ~available, 'Fewer than two shop receipts or no positive mean gap', 'flag')
        ratio = np.divide(rec, mean, out=np.zeros(len(ids)), where=available)
        f.add(f'sc_log_overdue_{suffix}', np.log1p(ratio), 'Log1p shop recency divided by shop-only mean gap', 'log_ratio')
        if suffix == 'life':
            lifetime_gap, lifetime_available = mean, available
    n = f.aligned(counts(shop))
    amount = f.aligned(shop.groupby('ID', observed=True).shop_scaled.sum()) / SCALE
    ticket = np.divide(amount, n, out=np.zeros(len(ids)), where=n > 0)
    horizon = float((t + pd.DateOffset(months=3) - t).days)
    denominator = np.maximum(lifetime_gap, 1)
    alive = np.exp(-np.maximum(rec - lifetime_gap, 0) / denominator)
    fills = horizon / denominator * alive
    fills[~lifetime_available] = 0
    z = np.log1p(np.maximum(fills * ticket, 0)) / config['normalization']['CLV_nonfuel']
    f.add('sc_corrected_cadence_z', z, 'Shop cadence proxy using shop gaps; zero when gap unavailable', 'transformed_nonfuel')
    f.add('sc_corrected_cadence_unavailable', ~lifetime_available, 'No positive measurable shop gap', 'flag')
    f.add('sc_net_rand_per_receipt', ticket, 'Signed lifetime nonfuel spend per shop receipt', 'rand')
    return f


def attachment_stats(f, events, suffix):
    alln = f.add(f'ba_all_n_{suffix}', counts(events), 'All receipt count in window')
    fuel = events[events.fuel_purchase]; shop = events[events.shop_purchase]
    mixed = events[events.mixed]
    fn = f.add(f'ba_fuel_n_{suffix}', counts(fuel), 'Qualifying fuel receipt count')
    sn = f.add(f'ba_shop_n_{suffix}', counts(shop), 'Qualifying shop receipt count')
    mn = f.add(f'ba_mixed_n_{suffix}', counts(mixed), 'Same-basket positive fuel and shop purchases')
    f.add(f'ba_qualifying_n_{suffix}', counts(events[events.qualifying_purchase]), 'Any in-scope qualifying purchase receipt')
    f.add(f'ba_fuel_only_n_{suffix}', counts(events[events.fuel_purchase & ~events.shop_purchase]), 'Fuel purchase without shop purchase')
    shoponly = events[events.shop_purchase & ~events.fuel_purchase]
    f.add(f'ba_shop_only_n_{suffix}', counts(shoponly), 'Shop purchase without qualifying fuel purchase')
    attach = f.ratio(f'ba_mixed_per_fuel_{suffix}', mn, fn, 'Share of qualifying fuel receipts with a shop purchase')
    f.ratio(f'ba_mixed_per_shop_{suffix}', mn, sn, 'Share of qualifying shop receipts with a fuel purchase')
    days = events.groupby(['ID', 'day'], observed=True)[['fuel_purchase', 'shop_purchase']].max()
    fd = days[days.fuel_purchase].groupby(level=0).size()
    md = days[days.fuel_purchase & days.shop_purchase].groupby(level=0).size()
    f.add(f'ba_same_day_both_n_{suffix}', md, 'Dates with qualifying fuel and shop purchases, possibly different receipts')
    f.ratio(f'ba_same_day_both_per_fuel_day_{suffix}', md, fd, 'Share of fuel-purchase dates also containing shop purchases')
    f.add(f'ba_mixed_shop_ticket_{suffix}', mixed.groupby('ID', observed=True).shop_scaled.mean() / SCALE, 'Signed shop rand per mixed receipt', 'rand')
    f.add(f'ba_shop_only_ticket_{suffix}', shoponly.groupby('ID', observed=True).shop_scaled.mean() / SCALE, 'Signed shop rand per shop-only receipt', 'rand')
    q = events[events.qualifying_purchase]
    f.add(f'ba_purchase_category_breadth_{suffix}', q.groupby('ID', observed=True).purchase_categories.mean(), 'Mean qualifying category breadth per qualifying receipt', 'mean_count')
    f.add(f'ba_purchase_product_breadth_{suffix}', q.groupby('ID', observed=True).purchase_products.mean(), 'Mean positive-purchase product breadth per qualifying receipt', 'mean_count')
    return attach


def basket_attachment(b, c, lines, t, ids, config, data_start):
    f = Features(ids)
    windows = [('30d', t-pd.Timedelta(days=30), t),
               ('q1', t-pd.DateOffset(months=3), t),
               ('q2', t-pd.DateOffset(months=6), t-pd.DateOffset(months=3))]
    rates = {}
    for suffix, start, end in windows:
        rates[suffix] = attachment_stats(f, in_window(b, start, end), suffix)
    f.add('ba_mixed_per_fuel_change', rates['q1']-rates['q2'], 'q1 minus q2 attachment rate; availability flags retained', 'ratio_change')
    mixed = b[b.mixed]
    f.add('ba_mixed_recency_days', recency(mixed, t), 'Days since last same-basket fuel/shop purchase', 'days', fill=999)
    f.add('ba_never_mixed', ~f.ids.isin(mixed.ID), 'No observed mixed basket', 'flag')
    return f


def slug(value):
    return value.lower().replace('&', 'and').replace(' ', '_').replace('-', '_').replace('/', '_').replace(':', '')


def category_state(b, c, lines, t, ids, config, data_start):
    f = Features(ids)
    included = sorted(k for k,v in config['category_mapping'].items() if v['included'])
    q1start, q2start = t-pd.DateOffset(months=3), t-pd.DateOffset(months=6)
    summary = {k: np.zeros(len(ids)) for k in ['ever', 'unseen', 'first_recent', 'reactivated', 'dormant', 'known_but_no_q1']}
    for cat in included:
        prefix = 'cs_' + slug(cat)
        rows = c[c.category == cat]
        purchases = rows[rows.purchase]
        before = purchases[purchases.time < q1start]
        recent = in_window(rows, q1start, t); previous = in_window(rows, q2start, q1start)
        recentp = recent[recent.purchase]; previousp = previous[previous.purchase]
        ever = f.add(prefix+'_ever', f.ids.isin(purchases.ID), 'Historical qualifying purchase in original '+cat, 'flag').astype(bool)
        known = f.add(prefix+'_before_q1', f.ids.isin(before.ID), 'Qualifying original-category purchase before q1', 'flag').astype(bool)
        n1 = f.add(prefix+'_q1_purchase_baskets', counts(recentp), 'Distinct positive qualifying category baskets in q1')
        n2 = f.add(prefix+'_q2_purchase_baskets', counts(previousp), 'Distinct positive qualifying category baskets in q2')
        a1 = f.add(prefix+'_q1_net_rand', recent.groupby('ID', observed=True).net_scaled.sum()/SCALE, 'Signed original-category q1 spend', 'rand')
        a2 = f.add(prefix+'_q2_net_rand', previous.groupby('ID', observed=True).net_scaled.sum()/SCALE, 'Signed original-category q2 spend', 'rand')
        for name, rows2 in [('q1',recentp),('q2',previousp)]:
            f.add(prefix+f'_{name}_purchase_days', rows2.groupby('ID', observed=True).day.nunique(), 'Distinct qualifying category purchase dates')
        f.add(prefix+'_recency_days', recency(purchases,t), 'Days since qualifying original-category purchase; ever flag disambiguates 999', 'days', fill=999)
        first = ~known & (n1>0)
        react = known & (n2==0) & (n1>0)
        dormant = (n2>0) & (n1==0)
        f.add(prefix+'_first_observed_recent',first,'First observed qualifying category purchase within q1','flag')
        f.add(prefix+'_reactivated',react,'Category known before q1, absent q2, purchased q1','flag')
        f.add(prefix+'_became_dormant',dormant,'Category purchased q2 but not q1','flag')
        delta = a1-a2
        f.add(prefix+'_signed_log_spend_change',np.sign(delta)*np.log1p(np.abs(delta)), 'Signed-log q1 minus q2 net spend','signed_log_rand')
        f.add(prefix+'_growth_absolute_spend_requirement',a1+config['growth_min_rand'], 'Known absolute future spend requirement from q1 and config, not a prediction','rand')
        relative = config['growth_min_relative']
        f.add(prefix+'_growth_relative_spend_boundary',np.maximum(a1,0)*(1+relative) if relative is not None else 0, 'Known relative spend boundary; executable label comparator remains authoritative','rand')
        if np.any((n1>0)&~ever) or np.any(known&~ever) or np.any(first&react):
            raise ValueError('inconsistent category states')
        for name, value in [('ever',ever),('unseen',~ever),('first_recent',first),('reactivated',react),('dormant',dormant),('known_but_no_q1',ever&(n1==0))]:
            summary[name] += value
    for name, value in summary.items():
        f.add('cs_categories_'+name, value, 'Count of original included categories in state '+name)
    if not np.all(summary['ever']+summary['unseen']==len(included)):
        raise ValueError('category breadth does not reconcile')
    return f


def coverage_and_ratios(b, c, lines, t, ids, config, data_start):
    f = Features(ids)
    first = b.groupby('ID', observed=True).time.min().reindex(f.ids)
    for k in range(1,5):
        start = t-pd.DateOffset(months=3*k);end=t-pd.DateOffset(months=3*(k-1))
        days=(end-start).total_seconds()/86400
        covered=max(0,(end-max(start,data_start)).total_seconds()/86400)
        f.add(f'cr_q{k}_dataset_coverage',min(covered/days,1),'Fraction of calendar window covered by released data','fraction')
        exposure_start=first.where(first>start,start).where(first.notna(),pd.NaT)
        exposure=((end-exposure_start).dt.total_seconds()/86400).clip(lower=0,upper=days).fillna(0)/days
        f.add(f'cr_q{k}_customer_observed_exposure',exposure,'Fraction of window after first observed customer receipt; not true tenure','fraction')
    for prefix,events in [('all',b),('fuel',b[b.fuel_any]),('shop',b[b.shop_any])]:
        stats=gap_summary(events)
        gap=f.aligned(stats['mean']);maximum=f.aligned(events.sort_values(['ID','time','TransactionId']).assign(
            _gap=lambda d:d.groupby('ID',observed=True).time.diff().dt.total_seconds()/86400).groupby('ID',observed=True)._gap.max())
        rec=f.aligned(recency(events,t),999)
        denoms=[('mean',gap),('max',maximum)] if prefix=='all' else [('mean',gap)]
        for kind,den in denoms:
            available=(den>0)
            value=np.divide(rec,den,out=np.zeros(len(ids)),where=available)
            f.add(f'cr_{prefix}_log_overdue_{kind}',np.log1p(value),'Log1p recency divided by positive historical '+kind+' gap','log_ratio')
            f.add(f'cr_{prefix}_gap_{kind}_unavailable',~available,'No positive gap denominator','flag')
        q1=f.aligned(counts(in_window(events,t-pd.DateOffset(months=3),t)))
        q2=f.aligned(counts(in_window(events,t-pd.DateOffset(months=6),t-pd.DateOffset(months=3))))
        f.add(f'cr_{prefix}_q2_empty',q2==0,'No receipts in prior quarter','flag')
        f.add(f'cr_{prefix}_resumed_from_empty_q2',(q1>0)&(q2==0),'Recent receipts following empty prior quarter, including first observed activity','flag')
        f.add(f'cr_{prefix}_became_empty_q1',(q2>0)&(q1==0),'Prior-quarter receipts but no recent receipts','flag')
        f.add(f'cr_{prefix}_log_count_change',np.log1p(q1)-np.log1p(q2),'Log1p q1 receipt count minus log1p q2 count','log_count_change')
    return f


def concentration(rows, key):
    n=rows.groupby(['ID',key],observed=True).size().rename('n').reset_index()
    if n.empty:
        return pd.DataFrame(columns=['breadth','share','entropy']),pd.Series(dtype=str)
    n['p']=n.n/n.groupby('ID',observed=True).n.transform('sum')
    n['ent']=-n.p*np.log(n.p)
    stats=n.groupby('ID',observed=True).agg(breadth=(key,'size'),share=('p','max'),entropy=('ent','sum'))
    # Deterministic lexical tie break, independent of input order.
    primary=n.sort_values(['ID','n',key],ascending=[True,False,True]).drop_duplicates('ID').set_index('ID')[key]
    return stats,primary


def fine_category_and_site(b,c,lines,t,ids,config,data_start):
    f=Features(ids)
    primaries={}
    for suffix,start,end in [('q1',t-pd.DateOffset(months=3),t),('q2',t-pd.DateOffset(months=6),t-pd.DateOffset(months=3))]:
        events=in_window(b,start,end)
        stats,primary=concentration(events,'site');primaries[suffix]=primary
        for key in ['breadth','share','entropy']:
            f.add(f'fs_site_{key}_{suffix}',stats[key],'Personal site '+key+' in '+suffix,'count' if key=='breadth' else 'ratio')
        q=lines[(lines._time>=start)&(lines._time<end)&lines._purchase][['ID','TransactionId','ItemCategoryLevel3']].copy()
        q['ItemCategoryLevel3']=q.ItemCategoryLevel3.astype(str)
        q=q.drop_duplicates(['ID','TransactionId','ItemCategoryLevel3'])
        fine,_=concentration(q,'ItemCategoryLevel3')
        for key in ['breadth','share','entropy']:
            f.add(f'fs_level3_{key}_{suffix}',fine[key],'Level3 qualifying category-basket '+key+' in '+suffix,'count' if key=='breadth' else 'ratio')
        f.add(f'fs_no_receipts_{suffix}',~f.ids.isin(events.ID),'No receipts for site comparison','flag')
    p1=primaries['q1'].reindex(f.ids);p2=primaries['q2'].reindex(f.ids)
    available=p1.notna()&p2.notna()
    f.add('fs_primary_site_changed',(p1!=p2)&available,'Dominant personal site differs between q2 and q1','flag')
    f.add('fs_primary_site_comparison_unavailable',~available,'At least one quarter lacks a dominant site','flag')
    start=t-pd.DateOffset(months=3)
    recent=b[b.time>=start].copy();past=b[b.time<start][['ID','site']].drop_duplicates()
    seen=past.assign(_seen=1)
    recent=recent.merge(seen,on=['ID','site'],how='left',validate='many_to_one')
    recent['_new_site']=recent._seen.isna()
    f.add('fs_new_to_history_site_share',recent.groupby('ID',observed=True)._new_site.mean(),'Fraction of q1 receipts at a site unseen before q1','ratio')
    return f


BUILDERS={name:globals()[name] for name in BLOCKS}


def build_customer_blocks(lines,baskets,categories,cutoff,ids,config,data_start):
    t=pd.Timestamp(cutoff)
    # Mandatory boundary on all three input layers, before feature aggregation.
    b=baskets[(baskets.time<t)&baskets.ID.isin(ids)]
    c=categories[(categories.time<t)&categories.ID.isin(ids)]
    d=lines[(lines._time<t)&lines.ID.isin(ids)]
    return {name:fn(b,c,d,t,ids,config,data_start) for name,fn in BUILDERS.items()}


def write_block(path,features,identity):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame=features.frame()
    values=frame.to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('float32 conversion lost finite values')
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,X=values,ids=frame.index.to_numpy(dtype=str),
                            feature_names=np.asarray(frame.columns,dtype=str),identity=identity)
    temp.replace(path)
    return {'rows':len(frame),'features':len(frame.columns),'nonfinite_cells':0,
            'constant_features':int((frame.nunique()==1).sum()),'sha256':digest(path)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out',type=Path)
    parser.add_argument('--phase',choices=['all','lines','baskets','features'],default='all')
    args=parser.parse_args()
    root=args.root.resolve();out=(args.out or root/'outputs/feature_rebuild').resolve()
    out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root/'src'))
    from reference import label_rules as lr
    config=json.loads((root/'configs/label_config.json').read_text())
    sources={name:digest(root/name) for name in ['data/raw/train.csv','data/raw/test.csv',
        'configs/label_config.json','src/reference/label_rules.py',
        'outputs/oct10/category_dynamics_panel.pkl','submissions/sasol_sprint16_r4full_vR.csv']}
    inputs={'version':VERSION,'source_hashes':sources,'builder_sha256':digest(Path(__file__))}
    identity=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    manifest_path=out/'construction_manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if previous['identity']!=identity:
            raise RuntimeError('output belongs to another source/code identity; use a new output directory')
    else:
        write_json(manifest_path,{'identity':identity,**inputs})
    write_json(out/'construction_proposal.json',construct_spec(config))
    started=time.monotonic()
    lpath=out/'layers/lines.pkl';bpath=out/'layers/baskets.pkl';cpath=out/'layers/category_baskets.pkl'
    # Record each cache checksum and verify before trusting a resumed artifact.
    checksums_path=out/'layer_checksums.json'
    checksums=json.loads(checksums_path.read_text()) if checksums_path.exists() else {}
    def cached(path):
        if not path.exists():
            return None
        name=str(path.relative_to(out))
        if name not in checksums or digest(path)!=checksums[name]:
            raise RuntimeError(f'unverified or corrupt layer cache: {path}')
        return pd.read_pickle(path)
    def store(path,value):
        write_pickle(path,value)
        checksums[str(path.relative_to(out))]=digest(path)
        write_json(checksums_path,checksums)
    lines=cached(lpath)
    if lines is None:
        log('Layer 1: authoritative transaction parsing and typed line flags')
        lines=enrich_lines(lr.load_transactions(root/'data/raw/train.csv'),config)
        store(lpath,lines)
    log(f'Layer 1 ready: {len(lines):,} lines, {lines.memory_usage(deep=True).sum()/1e6:.1f} MB in memory')
    layer1={'rows':len(lines),'columns':list(lines.columns),'duplicate_keys':int(lines.duplicated(KEY+['LineItemSequence']).sum()),
            'negative_amount_lines':int((lines._amount<0).sum()),'literal_NA_lines':int(lines.ItemCategoryLevel2.eq('NA').sum()),
            'data_start':str(lines._time.min()),'data_end':str(lines._time.max()),'identity':identity}
    write_json(out/'layer1_report.json',layer1)
    if args.phase=='lines':return
    b=cached(bpath);c=cached(cpath)
    if b is None or c is None:
        log('Layer 2: one row per basket and per original-category basket')
        b,c,checks=build_baskets(lines)
        store(bpath,b);store(cpath,c)
        write_json(out/'layer2_report.json',{'baskets':len(b),'category_baskets':len(c),'checks':checks})
    log(f'Layer 2 ready: {len(b):,} baskets, {len(c):,} original-category baskets')
    if args.phase=='baskets':return
    with (root/'outputs/oct10/category_dynamics_panel.pkl').open('rb') as f:
        panel=pickle.load(f)
    data_start=lines._time.min().floor('D')
    schemas={};reports={};leakage={}
    # Boundary verification uses original authoritative eligibility, never target labels.
    for cutoff,base in sorted(panel['frames'].items()):
        t=pd.Timestamp(cutoff);ids=base.index.astype(str)
        if not ids.is_unique:raise ValueError('duplicate panel IDs')
        eligible=set(lr.eligible_ids(lines,cutoff,config['minimum_historical_transactions']))
        if cutoff=='2025-12-01':
            test=pd.read_csv(root/'data/raw/test.csv',dtype=str,keep_default_na=False).ID.tolist()
            if len(ids)!=5488 or list(ids)!=test or not set(ids)<=eligible:
                raise ValueError('test ID order or eligibility mismatch')
        elif set(ids)!=eligible:
            raise ValueError(f'eligible ID set mismatch {cutoff}')
        blocks=build_customer_blocks(lines,b,c,cutoff,ids,config,data_start)
        reports[cutoff]={}
        for name,features in blocks.items():
            if name in schemas and list(schemas[name])!=list(features.catalog):
                raise ValueError(f'feature order changed {name} {cutoff}')
            schemas[name]=features.catalog
            reports[cutoff][name]=write_block(out/'features'/name/f'{cutoff}.npz',features,identity)
        log(f'Layer 3 {cutoff}: {len(ids)} customers, '+', '.join(f'{name}={len(x.values)}' for name,x in blocks.items()))
        if cutoff in ['2025-03-01','2025-09-01','2025-12-01']:
            # Rebuild baskets AFTER removing every future line. This detects future
            # contamination in global group aggregations, not just array slicing.
            truncated=lines[lines._time<t].copy()
            bt,ct,_=build_baskets(truncated)
            again=build_customer_blocks(truncated,bt,ct,cutoff,ids,config,data_start)
            for name in BLOCKS:
                pd.testing.assert_frame_equal(blocks[name].frame(),again[name].frame(),check_exact=True)
            leakage[cutoff]='All five blocks exactly equal after deleting future lines and rebuilding receipts'
            log(f'  PASS future-line deletion and receipt rebuild: {cutoff}')
    write_json(out/'feature_catalog.json',schemas)
    write_json(out/'feature_build_report.json',{'identity':identity,'cutoffs':reports,
        'future_line_deletion_checks':leakage,'features_per_block':{k:len(v) for k,v in schemas.items()},
        'baseline_features':len(panel['feature_cols']),'candidate_fits':0,'seconds':round(time.monotonic()-started,1)})
    # Check accepted source artifacts were not touched by this pipeline.
    for name,expected in sources.items():
        if digest(root/name)!=expected:raise RuntimeError(f'source changed during feature construction: {name}')
    write_json(out/'decision.json',{'construction_passed':True,'identity':identity,
        'source_hashes_unchanged':True,'future_line_deletion_checked_origins':list(leakage),
        'predictive_acceptance':'Not evaluated; no models fitted or submissions exported.'})
    log(f'COMPLETE: {out}; construction passed, no predictive claims')


if __name__=='__main__':
    main()
