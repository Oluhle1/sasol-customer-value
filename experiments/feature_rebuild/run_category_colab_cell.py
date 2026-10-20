from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
import json
import subprocess
import sys
import zipfile

base = Path('/content/drive/MyDrive/sasol')
archive_path = base / 'colab_category_state_bundle.zip'
folder = base / 'category_state_bundle'
assert archive_path.exists(), f'Upload the ZIP here first: {archive_path}'

with zipfile.ZipFile(archive_path) as archive:
    archive.extractall(folder)

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', '-q',
    'lightgbm==4.6.0', 'numpy==2.4.4', 'scikit-learn==1.8.0'
])
subprocess.run([
    sys.executable, 'colab_category_state.py', '--check'
], cwd=folder, check=True)

with (base / 'category_state_run.log').open('a') as log_file:
    with subprocess.Popen(
        [sys.executable, '-u', 'colab_category_state.py'],
        cwd=folder, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    ) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
            log_file.flush()
        code = process.wait()

if code:
    raise RuntimeError(f'Run stopped with exit code {code}; see the output above.')

result = folder / 'category_state_decision.json'
print(json.dumps(json.loads(result.read_text()), indent=2))
