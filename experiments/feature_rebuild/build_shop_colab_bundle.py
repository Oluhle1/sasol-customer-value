"""Build a self-contained Colab experiment from frozen artifacts; no fitting."""
import argparse
import hashlib
import json
import pickle
import tempfile
import zipfile
from pathlib import Path
import numpy as np

CELLS=[
    "from google.colab import drive\ndrive.mount('/content/drive')",
    "from pathlib import Path\nimport zipfile\nbase = Path('/content/drive/MyDrive/sasol')\nbundle = base / 'colab_shop_cadence_bundle.zip'\nassert bundle.exists(), f'Upload the ZIP here first: {bundle}'\nwith zipfile.ZipFile(bundle) as archive:\n    archive.extractall(base / 'shop_cadence_bundle')",
    "%pip install -q lightgbm==4.6.0 numpy==2.4.4 scikit-learn==1.8.0",
    "!cd /content/drive/MyDrive/sasol/shop_cadence_bundle && python colab_shop_cadence.py --check",
    "%%bash\nset -eo pipefail\ncd /content/drive/MyDrive/sasol/shop_cadence_bundle\npython -u colab_shop_cadence.py 2>&1 | tee -a ../shop_cadence_run.log",
    "from pathlib import Path\npath = Path('/content/drive/MyDrive/sasol/shop_cadence_bundle/shop_cadence_decision.json')\nprint(path.read_text() if path.exists() else 'No decision file yet. Run the training cell first.')"
]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--sources',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--out',type=Path);args=parser.parse_args()
    root=args.root;src=args.sources;out=args.out or root/'outputs/feature_rebuild'
    out.mkdir(parents=True,exist_ok=True)
    proposal=src/'proposal_shop_cadence.json'
    if not proposal.exists():proposal=root/'outputs/feature_rebuild/proposal_shop_cadence.json'
    with tempfile.TemporaryDirectory(prefix='sasol-shop-colab-') as temp:
        stage=Path(temp)
        with zipfile.ZipFile(root/'outputs/sprint17/colab_bundle.zip') as archive:
            for name in archive.namelist():
                if name.startswith(('probs/','rank4/')) or name in ['pool_r4full_vector.json','sasol_sprint16_r4full_vR.csv','colab_tabpfn.py']:
                    archive.extract(name,stage)
        # The neural bundle used float32. Tree models need original float64
        # values to avoid changing histogram splits in the baseline representation.
        with (root/'outputs/oct10/category_dynamics_panel.pkl').open('rb') as f:panel=pickle.load(f)
        plain={'feature_cols':list(panel['feature_cols']),'monthly_cutoffs':list(panel['monthly_cutoffs']),'cutoffs':{}}
        for cutoff,frame in panel['frames'].items():
            labels=panel['labels'].get(cutoff)
            plain['cutoffs'][cutoff]={'X':frame[panel['feature_cols']].to_numpy(np.float64),
                'ids':frame.index.to_numpy(dtype=str),
                'y':labels.reindex(frame.index).Opportunity.tolist() if labels is not None else None}
        with (stage/'panel_plain.pkl').open('wb') as f:pickle.dump(plain,f,protocol=4)
        features=root/'outputs/feature_rebuild'
        for name in ['feature_build_report.json','construction_manifest.json']:
            target=stage/'feature_layers'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((features/name).read_bytes())
        for cutoff in plain['cutoffs']:
            name=f'features/shop_cadence/{cutoff}.npz';target=stage/'feature_layers'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((features/name).read_bytes())
        (stage/'load_blocks.py').write_bytes((src/'load_blocks.py').read_bytes() if (src/'load_blocks.py').exists() else (root/'experiments/feature_rebuild/load_blocks.py').read_bytes())
        (stage/'colab_shop_cadence.py').write_bytes((src/'colab_shop_cadence.py').read_bytes())
        (stage/'proposal_shop_cadence.json').write_bytes(proposal.read_bytes())
        manifest={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'shop_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'Sasol_Shop_Cadence.ipynb'},'kernelspec':{'name':'python3','display_name':'Python 3'}},
            'cells':[{'cell_type':'markdown','metadata':{},'source':['# Shop cadence feature experiment\nUpload colab_shop_cadence_bundle.zip to MyDrive/sasol. Run cells in order in a fresh notebook. CPU runtime is sufficient.\nThis tests only 21 extra features; later folds and export run only if gates pass.']}]+
            [{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':[s+'\n' for s in text.splitlines()]} for text in CELLS]}
        body=json.dumps(notebook,indent=2)+'\n'
        (stage/'Sasol_Shop_Cadence.ipynb').write_text(body);(out/'Sasol_Shop_Cadence.ipynb').write_text(body)
        (stage/'README.txt').write_text('Shop cadence Colab experiment, baseline 305 -> 326 features.\nCPU runtime is sufficient. Use a fresh notebook.\nUpload this archive to MyDrive/sasol and open the companion notebook.\nNo need to upload feature_blocks_colab.zip separately.\nReturn shop_cadence_decision.json after the fit cell finishes.\n\n'+'\n\n'.join(CELLS)+'\n')
        target=out/'colab_shop_cadence_bundle.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob('*')):
                if path.is_file():archive.write(path,path.relative_to(stage))
        print(f'{target}: {target.stat().st_size/1e6:.2f} MB')


if __name__=='__main__':main()
