"""E14: frozen GPU FT-Transformer experiment. Run --check before fitting."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from colab_tabpfn import (POOL, EVAL, TEST, CLASSES, BASELINE_GUARD,
                         collapse_to_codes, combine, log)

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload, indent=2) + '\n')
    temp.replace(path)


def inputs():
    manifest = json.loads((HERE / 'manifest_fttransformer.json').read_text())
    for name, digest in manifest.items():
        if sha(HERE / name) != digest:
            raise RuntimeError(f'input hash mismatch: {name}')
    proposal = json.loads((HERE / 'proposal_fttransformer.json').read_text())
    with (HERE / 'panel_plain.pkl').open('rb') as f:
        panel = pickle.load(f)
    if len(panel['feature_cols']) != 305:
        raise RuntimeError('expected frozen 305 features')
    v = np.asarray(json.loads((HERE / 'pool_r4full_vector.json').read_text())['vR'])
    return proposal, panel, v


def load(kind, cutoff):
    p = np.load(HERE / kind / f'{cutoff}.npz', allow_pickle=False)['proba'].astype(float)
    validate(p, 17 if kind == 'probs' else 4)
    return p


def validate(p, columns):
    if p.ndim != 2 or p.shape[1] != columns or not np.isfinite(p).all() or (p < 0).any():
        raise RuntimeError('invalid probability matrix')
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise RuntimeError('probability rows do not sum to one')


def score(panel, cutoff, p4, v):
    pred = np.asarray(CLASSES)[np.argmax(combine(load('probs', cutoff), p4) * v, axis=1)]
    return float(f1_score(panel['cutoffs'][cutoff]['y'], pred, average='weighted', zero_division=0))


def shift(cutoff, months):
    return str(np.datetime64(cutoff, 'M') + months) + '-01'


def split(panel, cutoff):
    latest = shift(cutoff, -3)
    limit = shift(latest, -3)
    train = [o for o in panel['monthly_cutoffs'] if o <= limit and panel['cutoffs'][o]['y'] is not None]
    if not train or panel['cutoffs'][latest]['y'] is None:
        raise RuntimeError(f'no purged inner split for {cutoff}')
    if not all(shift(o, 3) <= latest for o in train):
        raise RuntimeError('inner label overlap')
    def arrays(origins):
        x = np.concatenate([panel['cutoffs'][o]['X'] for o in origins]).astype(np.float32)
        y = np.concatenate([collapse_to_codes(panel['cutoffs'][o]['y']) for o in origins])
        weights = []
        for o in origins:
            year, month = map(int, o[:7].split('-'))
            weight = .70 + .05 * ((year - 2024) * 12 + month - 9)
            weights.extend([weight] * len(panel['cutoffs'][o]['X']))
        return x, y, np.asarray(weights, dtype=np.float32)
    return arrays(train), arrays([latest]), train, latest


class FitTimeout(Exception):
    pass


def fit(panel, cutoff, proposal, identity):
    import torch
    import torch.nn.functional as F
    from rtdl_revisiting_models import FTTransformer

    dest = HERE / 'fttransformer'
    dest.mkdir(exist_ok=True)
    path = dest / f'{cutoff}_fttransformer.npz'
    ids = panel['cutoffs'][cutoff]['ids'].astype(str)
    if path.exists():
        with np.load(path, allow_pickle=False) as data:
            if str(data['identity']) != identity or not np.array_equal(data['ids'], ids):
                raise RuntimeError(f'incompatible cache: {path}')
            p = data['proba'].astype(float)
            validate(p, 4)
            if len(p) != len(ids):
                raise RuntimeError('cached row count mismatch')
            return p, json.loads(str(data['metadata']))
    started = time.monotonic()
    def deadline():
        if time.monotonic() - started > proposal['fit_timeout_seconds']:
            raise FitTimeout(f'{cutoff} exceeded fixed fit budget')
    (x, y, w), (xv, yv, _), origins, latest = split(panel, cutoff)
    # Only inner training rows determine imputation and scaling.
    x = np.where(np.isfinite(x), x, np.nan)
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0).astype(np.float32)
    x = np.where(np.isfinite(x), x, median)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd < 1e-6, 1, sd)
    def standardize(a):
        a = np.where(np.isfinite(a), a, median)
        return torch.from_numpy(np.clip((a - mu) / sd, -8, 8).astype(np.float32)).cuda()
    xt, xv = standardize(x), standardize(xv)
    xp = standardize(panel['cutoffs'][cutoff]['X'])
    yt = torch.from_numpy(y).cuda()
    yv = torch.from_numpy(yv).cuda()
    wt = torch.from_numpy(w).cuda()
    torch.manual_seed(proposal['seed'])
    torch.cuda.manual_seed_all(proposal['seed'])
    np.random.seed(proposal['seed'])
    model = FTTransformer(n_cont_features=xt.shape[1], cat_cardinalities=[], d_out=4,
                          **proposal['architecture']).cuda()
    optimizer = model.make_default_optimizer()
    # Explicitly freeze the optimizer settings, including its no-decay groups.
    for group in optimizer.param_groups:
        group['lr'] = proposal['learning_rate']
        if group['weight_decay']:
            group['weight_decay'] = proposal['weight_decay']
    batch = proposal['batch_size']
    def logits(a):
        return model(a, None)
    def validation_loss():
        total = 0.0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(xv), batch):
                deadline()
                total += float(F.cross_entropy(logits(xv[i:i+batch]), yv[i:i+batch], reduction='sum'))
        return total / len(xv)
    best, state, stale, best_epoch = float('inf'), None, 0, None
    for epoch in range(proposal['max_epochs']):
        model.train()
        perm = torch.randperm(len(xt), device='cuda')
        for i in range(0, len(xt), batch):
            deadline()
            idx = perm[i:i+batch]
            optimizer.zero_grad(set_to_none=True)
            losses = F.cross_entropy(logits(xt[idx]), yt[idx], reduction='none')
            loss = (losses * wt[idx]).sum() / wt[idx].sum()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        value = validation_loss()
        if not np.isfinite(value):
            raise RuntimeError('nonfinite validation loss')
        if value < best - 1e-5:
            best, stale, best_epoch = value, 0, epoch + 1
            state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        else:
            stale += 1
        log(f'FT {cutoff} epoch {epoch+1}: inner logloss {value:.5f}, best {best:.5f}, elapsed {time.monotonic()-started:.0f}s')
        if stale >= proposal['patience']:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        chunks = []
        for i in range(0, len(xp), batch):
            deadline()
            chunks.append(logits(xp[i:i+batch]).softmax(-1).cpu().numpy())
        p = np.concatenate(chunks)
    validate(p, 4)
    meta = {'train_origins': origins, 'inner_validation_origin': latest,
            'train_rows': len(x), 'validation_rows': len(xv), 'best_epoch': best_epoch,
            'inner_logloss': best, 'seconds': round(time.monotonic()-started, 1)}
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, proba=p, ids=ids, identity=identity, metadata=json.dumps(meta))
    temp.replace(path)
    del model, optimizer, xt, xv, xp, yt, yv, wt
    torch.cuda.empty_cache()
    return p, meta


def export(panel, p4, v, decision):
    pred = np.asarray(CLASSES)[np.argmax(combine(load('probs', TEST), p4) * v, axis=1)]
    source = HERE / 'sasol_sprint16_r4full_vR.csv'
    with source.open(newline='') as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    if header != ['ID', 'CLV_fuel', 'CLV_nonfuel', 'Opportunity']:
        raise RuntimeError('unexpected submission header')
    ids = [r[0] for r in rows]
    if len(ids) != 5488 or len(set(ids)) != 5488 or ids != panel['cutoffs'][TEST]['ids'].astype(str).tolist():
        raise RuntimeError('official IDs/order mismatch')
    clv = np.asarray([[float(r[1]), float(r[2])] for r in rows])
    if not np.isfinite(clv).all() or (clv < 0).any():
        raise RuntimeError('invalid source CLV')
    path = HERE / 'sasol_sprint17_fttransformermix.csv'
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([r[:3] + [str(p)] for r, p in zip(rows, pred)])
    with path.open(newline='') as f:
        check = list(csv.reader(f))[1:]
    if any(a[:3] != b[:3] for a, b in zip(rows, check)):
        raise RuntimeError('source ID/CLV strings changed')
    decision['exported'] = {'file': path.name, 'rows': len(rows),
                            'source_sha256': sha(source), 'sha256': sha(path),
                            'changes_vs_best': sum(r[3] != p for r, p in zip(rows, pred))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true', help='verify bundle and splits without fitting')
    args = parser.parse_args()
    proposal, panel, v = inputs()
    # Check every baseline before spending GPU time. No candidate evaluation here.
    baseline = {}
    for c in POOL + EVAL:
        baseline[c] = score(panel, c, load('rank4', c), v)
        if abs(baseline[c] - BASELINE_GUARD[c]) > 1e-9:
            raise RuntimeError(f'baseline mismatch: {c}')
        train, validation, origins, latest = split(panel, c)
        log(f'CHECK {c}: baseline {baseline[c]:.6f}, train {len(train[0])}, inner {latest} ({len(validation[0])}), last train {origins[-1]}')
    if args.check:
        log('PASS: input hashes, seven baselines and purged splits; no fits run')
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab; CPU fitting is disabled')
    started = time.monotonic()
    versions = {name: importlib.metadata.version(name) for name in ['torch', 'numpy', 'scikit-learn', 'rtdl_revisiting_models']}
    identity = hashlib.sha256(((HERE / 'manifest_fttransformer.json').read_text() + json.dumps(versions, sort_keys=True)).encode()).hexdigest()
    decision = {'experiment': proposal['name'], 'proposal_sha256': sha(HERE / 'proposal_fttransformer.json'),
                'identity': identity, 'versions': versions, 'python': platform.python_version(),
                'device': torch.cuda.get_device_name(), 'baseline': baseline, 'members': {}}
    path = HERE / 'fttransformer_decision.json'
    save_json(path, decision)
    def run(cutoff):
        p, meta = fit(panel, cutoff, proposal, identity)
        decision['members'][cutoff] = meta
        return .5 * load('rank4', cutoff) + .5 * p
    def stage(cutoffs, gate):
        scores, gains = {}, {}
        for c in cutoffs:
            scores[c] = score(panel, c, run(c), v)
            gains[c] = scores[c] - baseline[c]
            log(f'FT-mix {c}: {scores[c]:.6f}, gain {gains[c]:+.6f}')
            save_json(path, decision)
        mean, minimum = float(np.mean(list(gains.values()))), min(gains.values())
        return {'scores': scores, 'gains': gains, 'mean_gain': mean, 'min_gain': minimum,
                'passed': mean >= gate['mean_gain_min'] and minimum >= gate['per_origin_min']}
    try:
        decision['screen'] = stage(POOL, proposal['screen_gate'])
        save_json(path, decision)
        if decision['screen']['passed']:
            decision['folds'] = stage(EVAL, proposal['eval_gate'])
            save_json(path, decision)
            if decision['folds']['passed']:
                export(panel, run(TEST), v, decision)
    except FitTimeout as e:
        decision['aborted'] = str(e)
    except Exception as e:
        decision['error'] = f'{type(e).__name__}: {e}'
        save_json(path, decision)
        raise
    finally:
        decision['seconds'] = round(time.monotonic()-started, 1)
        save_json(path, decision)
    print('\n===== COPY THIS BACK =====')
    print(json.dumps(decision, indent=2))
    print('Send back fttransformer_decision.json and fttransformer/*.npz' +
          (', ' + decision['exported']['file'] if 'exported' in decision else ''))


if __name__ == '__main__':
    main()
