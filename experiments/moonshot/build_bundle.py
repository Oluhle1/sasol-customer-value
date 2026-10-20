"""Package fixed moonshot experiments without fitting or rebuilding features."""
import argparse,hashlib,json,tempfile,zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);parser.add_argument('--out',type=Path);args=parser.parse_args()
    root=args.root;source=Path(__file__).resolve().parent;out=args.out or root/'outputs/moonshot';out.mkdir(parents=True,exist_ok=True)
    proposal=source/'proposal.json'
    if not proposal.exists():proposal=root/'outputs/moonshot/proposal.json'
    with tempfile.TemporaryDirectory(prefix='sasol-moonshot-') as tmp:
        stage=Path(tmp)
        with zipfile.ZipFile(root/'outputs/feature_rebuild/colab_shop_cadence_bundle.zip') as z:
            for name in z.namelist():
                if name.startswith(('probs/','rank4/')) or name in ['panel_plain.pkl','colab_tabpfn.py','load_blocks.py','pool_r4full_vector.json','sasol_sprint16_r4full_vR.csv']:z.extract(name,stage)
        with zipfile.ZipFile(root/'outputs/feature_rebuild/feature_blocks_colab.zip') as z:
            for name in z.namelist():
                if name.startswith('features/') or name in ['construction_manifest.json','feature_build_report.json']:
                    path=stage/'feature_layers'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
        (stage/'label_config.json').write_bytes((root/'configs/label_config.json').read_bytes())
        (stage/'moonshot.py').write_bytes((source/'moonshot.py').read_bytes())
        (stage/'proposal.json').write_bytes(proposal.read_bytes())
        manifest={str(p.relative_to(stage)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        target=out/'sasol_moonshot_bundle.zip'
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob('*')):
                if p.is_file():z.write(p,p.relative_to(stage))
        print(f'{target}: {target.stat().st_size/1e6:.2f} MB')


if __name__=='__main__':main()
