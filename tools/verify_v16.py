"""Verify v16 publication identities and unchanged historical blobs, stdlib only.

The original verify_release.py intentionally retains its initial README hashes.
This revision-aware verifier checks the current docs against the new manifest
and every earlier published file against the recorded base Git blob identity.
It does not open original images, checkpoint files or prediction databases.
"""
from pathlib import Path,PurePosixPath
import hashlib,json

ROOT=Path(__file__).resolve().parents[1]

def need(value,message):
    if not value:raise ValueError(message)

def safe(root,name):
    rel=PurePosixPath(name)
    need(not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'Unsafe manifest path')
    p=root.joinpath(*rel.parts).resolve();need(root.resolve() in p.parents and p.is_file(),'Missing manifest file: '+name)
    return p

def gitsha(data):return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()

def verify_files(root):
    root=Path(root)
    old=json.loads((root/'manifests/v16_base_blobs.json').read_bytes())
    current=json.loads((root/'manifests/v16_extension_files.json').read_bytes())
    replacements=old['replaced_documentation_paths']
    need(sorted(replacements)==['README.md','docs/RELEASE_STATUS.md'],'Unexpected historical replacement scope')
    preserved=0
    for name,h in old['base_blob_sha'].items():
        if name in replacements:
            need(name in current,'Current documentation identity absent');continue
        need(gitsha(safe(root,name).read_bytes())==h,'Historical Git blob differs: '+name);preserved+=1
    for name,meta in current.items():
        data=safe(root,name).read_bytes()
        need(len(data)==meta['bytes'] and hashlib.sha256(data).hexdigest()==meta['sha256'] and gitsha(data)==meta['git_blob_sha'],'v16 file identity differs: '+name)
    return {'historical_blobs_preserved':preserved,'current_files_verified':len(current),'replaced_docs_verified':len(replacements)}

def verify_roles(root):
    records=json.loads((root/'splits/image_identities.json').read_bytes())['records']
    need(len(records)==len({r['source_image'] for r in records})==7019,'Original source coverage differs')
    roles={}
    for role,count in [('training',5824),('development',647),('official_val',548)]:
        names=(root/'splits'/(role+'.txt')).read_text().splitlines()
        need(len(names)==count and names==sorted(r['source_image'] for r in records if r['role']==role),'Role membership differs: '+role)
        roles[role]=set(names)
    need(not any(roles[a]&roles[b] for a,b in [('training','development'),('training','official_val'),('development','official_val')]),'Roles overlap')
    grouped={}
    for r in records:grouped.setdefault(r['rgb_sha256'],set()).add(r['role'])
    need(not any(len(v)>1 for v in grouped.values()),'Recorded exact RGB identity crosses roles')
    return {k:len(v) for k,v in roles.items()}

def main():
    result=verify_files(ROOT);result.update({'status':'PASS','recorded_roles':verify_roles(ROOT),
          'scope':'publication file identities and recorded roles; no original images, weights, predictions or model inference opened'})
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
