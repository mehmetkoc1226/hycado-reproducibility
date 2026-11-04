import hashlib,json,tempfile,unittest,zipfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import stage_external_artifacts as s

class StageTest(unittest.TestCase):
 def test_missing_corrupt_atomic_and_replay(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);project=root/'p';inputs=root/'i';out=root/'out';man=root/'m';inputs.mkdir();man.mkdir()
   cp={};files={}
   for key,(directory,_) in s.CHECKPOINTS.items():
    p=project/'experiments'/directory/'weights'/'epoch_0080.pt';p.parent.mkdir(parents=True)
    p.write_bytes((key+' model').encode());cp[key]={'final_sha256':s.digest(p)}
   for i in range(5):
    p=inputs/(str(i)+'.zip')
    with zipfile.ZipFile(p,'w') as z:z.writestr('item.txt',str(i))
    files[p.name]=s.digest(p)
   (man/'checkpoints.json').write_text(json.dumps(cp));(man/'global_fusion_inputs.json').write_text(json.dumps({'files':files}))
   p=inputs/'0.zip';good=p.read_bytes();p.write_bytes(b'bad')
   with self.assertRaises(ValueError):s.run(project,inputs,out,man,False)
   self.assertFalse(out.exists());p.write_bytes(good)
   s.run(project,inputs,out,man,True);s.verify_staged(out)
   sha=s.digest(out/'checkpoints/yolo_epoch_0080.pt');s.run(project,inputs,out,man,True)
   self.assertEqual(sha,s.digest(out/'checkpoints/yolo_epoch_0080.pt'))
   (out/'checkpoints/yolo_epoch_0080.pt').write_bytes(b'tampered')
   with self.assertRaises(ValueError):s.verify_staged(out)
   with self.assertRaises(ValueError):s.run(project,inputs,out,man,True)

 def test_nested_directory_and_unopened_bundle(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);names={'one.zip':'a','two.zip':'b','three.zip':'c','four.zip':'d','decision.md':'e'}
   nested=root/'unpacked'/'files';nested.mkdir(parents=True)
   expected={}
   for name,value in names.items():
    p=nested/name
    if name.endswith('.zip'):
     with zipfile.ZipFile(p,'w') as z:z.writestr('value.txt',value)
    else:p.write_text(value)
    expected[name]=s.digest(p)
   self.assertEqual(s.locate_inputs(root,expected)[0],nested)
   bundle=root/s.BUNDLE
   with zipfile.ZipFile(bundle,'w') as z:
    for name in names:z.write(nested/name,name)
   for file in nested.iterdir():file.unlink()
   actual_sha=s.BUNDLE_SHA256
   try:
    s.BUNDLE_SHA256=s.digest(bundle)
    folder,temp=s.locate_inputs(root,expected)
    try:
     self.assertEqual({n:s.digest(folder/n) for n in names},expected)
    finally:temp.cleanup()
   finally:s.BUNDLE_SHA256=actual_sha
   self.assertFalse(folder.exists())

if __name__=='__main__':unittest.main()
