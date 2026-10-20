"""Build E14 from the frozen E13 plain-array bundle. Never fits models."""
import argparse
import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


CELLS = [
    "from google.colab import drive\ndrive.mount('/content/drive')",
    "from pathlib import Path\nimport zipfile\nbase = Path('/content/drive/MyDrive/sasol')\nwith zipfile.ZipFile(base / 'colab_fttransformer_bundle.zip') as z:\n    z.extractall(base / 'fttransformer_bundle')",
    "%pip install -q rtdl_revisiting_models==0.0.2",
    "!cd /content/drive/MyDrive/sasol/fttransformer_bundle && python colab_fttransformer.py --check",
    "!set -o pipefail; cd /content/drive/MyDrive/sasol/fttransformer_bundle && python -u colab_fttransformer.py 2>&1 | tee -a /content/drive/MyDrive/sasol/fttransformer_run.log",
    "from pathlib import Path\nimport zipfile\nbase = Path('/content/drive/MyDrive/sasol/fttransformer_bundle')\nfiles = [base / 'fttransformer_decision.json', *sorted((base / 'fttransformer').glob('*.npz'))]\nsubmission = base / 'sasol_sprint17_fttransformermix.csv'\nif submission.exists():\n    files.append(submission)\nwith zipfile.ZipFile(base.parent / 'fttransformer_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n    for p in files:\n        if p.exists():\n            z.write(p, p.relative_to(base))\nprint(base.parent / 'fttransformer_results.zip')"
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--sources', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    out = args.output_dir or args.root / 'outputs/sprint17'
    out.mkdir(parents=True, exist_ok=True)
    proposal = args.sources / 'proposal_fttransformer.json'
    if not proposal.exists():
        proposal = args.root / 'outputs/sprint17/proposal_fttransformer.json'
    with tempfile.TemporaryDirectory(prefix='sasol-e14-') as temp:
        stage = Path(temp)
        with zipfile.ZipFile(args.root / 'outputs/sprint17/colab_bundle.zip') as archive:
            for name in archive.namelist():
                if name == 'README.txt':
                    continue
                if name.startswith('/') or '..' in Path(name).parts:
                    raise RuntimeError('unsafe archive path')
                archive.extract(name, stage)
        (stage / 'colab_fttransformer.py').write_bytes((args.sources / 'colab_fttransformer.py').read_bytes())
        (stage / 'proposal_fttransformer.json').write_bytes(proposal.read_bytes())
        manifest = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(stage.rglob('*')) if p.is_file()}
        (stage / 'manifest_fttransformer.json').write_text(json.dumps(manifest, indent=2) + '\n')
        notebook = {'nbformat': 4, 'nbformat_minor': 5,
                    'metadata': {'colab': {'name': 'Sasol_E14_FTTransformer.ipynb'},
                                 'accelerator': 'GPU', 'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}},
                    'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
                        '# Sasol E14: FT-Transformer\nUpload the ZIP to MyDrive/sasol. Select a T4 GPU runtime, then run cells in order.\nOnly the screen runs initially; later folds and export are gated automatically.\nRerun the fit cell after disconnects to reuse completed origins.']}] +
                    [{'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                      'source': [line + '\n' for line in cell.splitlines()]} for cell in CELLS]}
        notebook_text = json.dumps(notebook, indent=2) + '\n'
        (stage / 'Sasol_E14_FTTransformer.ipynb').write_text(notebook_text)
        (out / 'Sasol_E14_FTTransformer.ipynb').write_text(notebook_text)
        readme = ('Sasol E14 FT-Transformer GPU experiment\n\n'
                  'Upload colab_fttransformer_bundle.zip to MyDrive/sasol/.\n'
                  'Open Sasol_E14_FTTransformer.ipynb in Colab and select T4 GPU.\n'
                  'Run cells in order; completed folds survive in Drive.\n'
                  'Return fttransformer_results.zip or paste the decision JSON.\n'
                  'No public score is inferred from local results.\n\n' + '\n\n'.join(CELLS) + '\n')
        (stage / 'README.txt').write_text(readme)
        target = out / 'colab_fttransformer_bundle.zip'
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
            for p in sorted(stage.rglob('*')):
                if p.is_file():
                    archive.write(p, p.relative_to(stage))
        print(f'{target}: {target.stat().st_size / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
