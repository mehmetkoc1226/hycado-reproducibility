"""Validate and stage the exact historical checkpoint/prediction bytes.

Python standard library only. No torch, model load, training, or inference.
Does not upload or publish anything. All originals remain untouched.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,sys,zipfile
from pathlib import Path

CHECKPOINTS={
 'yolo':('stage1_v1_seed42','yolo_epoch_0080.pt'),
 'crop':('rtdetr_crop_v1_seed42','crop_epoch_0080.pt'),
 'full':('rtdetr_full_image_v1_seed42','full_epoch_0080.pt'),
}
BLOCK=4*1024*1024

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(BLOCK),b''):h.update(block)
 return h.hexdigest()

def verified(path,expected):
 if not path.is_file():raise FileNotFoundError(str(path))
 actual=digest(path)
 if actual!=expected:raise ValueError(f'Hash mismatch: {path.name}: {actual} != {expected}')
 return {'sha256':actual,'bytes':path.stat().st_size}

def check_zip(path):
 with zipfile.ZipFile(path) as z:
  bad=z.testzip()
  if bad:raise ValueError('Corrupt ZIP member: '+bad)

def sources(project,inputs,manifest):
 specs=[]
 for key,(directory,name) in CHECKPOINTS.items():
  entry=manifest['checkpoints'][key]
  specs.append((project/'experiments'/directory/'weights'/'epoch_0080.pt', 'checkpoints/'+name,entry['final_sha256'],False))
 for name,expected in manifest['inputs']['files'].items():
  specs.append((inputs/name,'prediction_inputs/'+name,expected,name.endswith('.zip')))
 return specs

def write_json(path,record):
 tmp=path.with_name(path.name+'.tmp')
 tmp.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 os.replace(tmp,path)

def run(project,inputs,out,manifest_dir,stage):
 m={'checkpoints':json.loads((manifest_dir/'checkpoints.json').read_text()),
    'inputs':json.loads((manifest_dir/'global_fusion_inputs.json').read_text())}
 if set(m['checkpoints'])!=set(CHECKPOINTS) or len(m['inputs']['files'])!=5:raise ValueError('Invalid fixed manifest')
 checked={};missing=[]
 for path,name,sha,is_zip in sources(project,inputs,m):
  try:
   row=verified(path,sha)
   if is_zip:check_zip(path)
   checked[name]=row
  except FileNotFoundError:missing.append(name)
 if missing:
  print(json.dumps({'status':'INCOMPLETE','verified':checked,'missing':missing},indent=2))
  raise ValueError('Missing required files: '+', '.join(missing))
 if not stage:
  print(json.dumps({'status':'VERIFIED','verified':checked,'staged':False},indent=2));return
 if project.resolve()==out.resolve() or inputs.resolve()==out.resolve():raise ValueError('Output cannot replace source root')
 out.mkdir(parents=True,exist_ok=True)
 for src,name,sha,_ in sources(project,inputs,m):
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():verified(dest,sha)
  else:
   tmp=dest.with_name(dest.name+'.partial')
   if tmp.exists():tmp.unlink() # Incomplete prior copy only; original is untouched.
   shutil.copyfile(src,tmp)
   verified(tmp,sha)
   os.replace(tmp,dest)
  print('STAGED:',name,flush=True)
 record={'status':'STAGED_VERIFIED','files':checked,
  'source_repository':'https://github.com/mehmetkoc1226/hycado-reproducibility',
  'scope':'exact bytes staged; no training, inference, or publication performed',
  'public_download_urls':None}
 write_json(out/'manifest.json',record)
 print('STAGED VERIFIED:',len(checked),'files; manifest:',out/'manifest.json')

def verify_staged(out):
 record=json.loads((out/'manifest.json').read_text())
 if record['status']!='STAGED_VERIFIED' or len(record['files'])!=8:raise ValueError('Invalid stage manifest')
 for name,item in record['files'].items():
  if name.startswith('/') or '..' in Path(name).parts or not (name.startswith('prediction_inputs/') or name.startswith('checkpoints/')):raise ValueError('Unsafe artifact path')
  actual=verified(out/name,item['sha256'])
  if actual['bytes']!=item['bytes']:raise ValueError('Size mismatch: '+name)
  if name.endswith('.zip'):check_zip(out/name)
 print('STAGED FILES VALID:',len(record['files']))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--project',type=Path);p.add_argument('--inputs',type=Path)
 p.add_argument('--output',type=Path);p.add_argument('--stage',action='store_true');p.add_argument('--verify-staged',action='store_true')
 p.add_argument('--manifest-dir',type=Path,default=Path(__file__).resolve().parents[1]/'manifests')
 a=p.parse_args()
 try:
  if a.verify_staged:
   if a.output is None:raise ValueError('--output required')
   verify_staged(a.output)
  else:
   if a.project is None or a.inputs is None:raise ValueError('--project and --inputs required')
   if a.stage and a.output is None:raise ValueError('--output required for --stage')
   run(a.project,a.inputs,a.output,a.manifest_dir,a.stage)
 except (ValueError,OSError,zipfile.BadZipFile) as exc:p.exit(1,str(exc)+'\n')
