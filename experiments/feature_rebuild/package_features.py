"""Package verified numeric feature blocks for Colab; no raw/pandas caches."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'outputs/feature_rebuild')
    args=parser.parse_args();root=args.out
    decision=json.loads((root/'decision.json').read_text())
    if not decision['construction_passed']:
        raise RuntimeError('construction has not passed')
    source=Path(__file__).resolve().parent
    report=json.loads((root/'feature_build_report.json').read_text())
    files={name:root/name for name in ['construction_manifest.json','construction_proposal.json',
        'feature_build_report.json','feature_catalog.json','decision.json','layer1_report.json','layer2_report.json']}
    for cutoff,blocks in report['cutoffs'].items():
        for name,meta in blocks.items():
            relative=f'features/{name}/{cutoff}.npz';path=root/relative
            if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:
                raise RuntimeError(f'corrupt block {path}')
            files[relative]=path
    for name in ['load_blocks.py','README.md']:
        files[name]=source/name
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    target=root/'feature_blocks_colab.zip';temp=target.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,path in sorted(files.items()):archive.write(path,name)
        archive.writestr('bundle_manifest.json',json.dumps(hashes,indent=2)+'\n')
    temp.replace(target)
    print(f'{target} ({target.stat().st_size/1e6:.2f} MB), {len(report["cutoffs"])} cutoffs, no raw/pandas layer files')


if __name__=='__main__':main()
