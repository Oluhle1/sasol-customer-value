from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import hashlib,json,subprocess,sys,zipfile
base=Path('/content/drive/MyDrive/sasol')
original=base/'sasol_moonshot_bundle.zip'
patch=base/'sasol_moonshot_cpu_patch.zip'
folder=base/'moonshot_cpu_memory_bundle'
assert original.exists() and patch.exists(),'Keep the original moonshot ZIP and upload the new CPU patch ZIP to MyDrive/sasol.'
with zipfile.ZipFile(patch) as update:
    parent=json.loads(update.read('parent.json'))
    with zipfile.ZipFile(original) as archive:
        assert hashlib.sha256(archive.read('manifest.json')).hexdigest()==parent['original_manifest_sha256'],'Original ZIP does not match the CPU patch.'
        archive.extractall(folder)
    update.extractall(folder)
subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==2.4.4','scikit-learn==1.8.0','lightgbm==4.6.0','catboost==1.2.10'])
subprocess.run([sys.executable,'moonshot.py','--check'],cwd=folder,check=True)
with (base/'moonshot_cpu_memory_run.log').open('a') as log_file:
    with subprocess.Popen([sys.executable,'-u','moonshot.py'],cwd=folder,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as process:
        for line in process.stdout:
            print(line,end='',flush=True);log_file.write(line);log_file.flush()
        code=process.wait()
if code:raise RuntimeError(f'Run stopped with exit code {code}; see output above')
print((folder/'moonshot_decision.json').read_text())
