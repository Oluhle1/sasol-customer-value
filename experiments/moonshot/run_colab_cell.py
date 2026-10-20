from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import json,subprocess,sys,zipfile
base=Path('/content/drive/MyDrive/sasol')
archive=base/'sasol_moonshot_bundle.zip'
folder=base/'moonshot_bundle'
assert archive.exists(),f'Upload the ZIP here first: {archive}'
with zipfile.ZipFile(archive) as z:z.extractall(folder)
subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==2.4.4','scikit-learn==1.8.0','lightgbm==4.6.0','catboost==1.2.10'])
subprocess.run([sys.executable,'-c','from catboost.utils import get_gpu_device_count; assert get_gpu_device_count()>0, "Select a T4 GPU runtime first"'],check=True)
subprocess.run([sys.executable,'moonshot.py','--check'],cwd=folder,check=True)
with (base/'moonshot_run.log').open('a') as log_file:
    with subprocess.Popen([sys.executable,'-u','moonshot.py'],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        for line in process.stdout:
            print(line,end='',flush=True)
            log_file.write(line);log_file.flush()
        code=process.wait()
if code:raise RuntimeError(f'Run stopped with exit code {code}; see output above')
result=json.loads((folder/'moonshot_decision.json').read_text())
print(json.dumps(result,indent=2))
print('CSV exports:',json.dumps(result['exports'],indent=2))
