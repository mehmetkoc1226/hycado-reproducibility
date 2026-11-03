import contextlib,copy,io,tempfile,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
from hycado_start import full_image_training as f
from hycado_start import crop_experiment as engine


class FullTrainingTests(unittest.TestCase):
    def test_whole_image_groups_and_fixed80_schedule(self):
        records=[{'source_image':str(i),'split':'train' if i<5824 else 'val'} for i in range(5856)]
        dev=[str(i) for i in range(5824,5856)]
        def sample(stage,data,r,signature):return {'source_image':r['source_image'],'split':r['split'],'crop_index':0}
        with tempfile.TemporaryDirectory() as tmp,patch.object(f.p,'materialize',side_effect=sample),contextlib.redirect_stdout(io.StringIO()):
            groups,items=f.prepare(Path(tmp),records,Path(tmp)/'data',{'fixed':True},dev)
        self.assertEqual(len(groups['train']),5824);self.assertEqual([x['source_image'] for x in items],dev)
        self.assertTrue(all(len(x['crops'])==1 for x in groups['train']))
        cfg={'rounds':80,'samples_per_round':5824,'seed':42,'lr':1e-4,'lrf':.1}
        schedule=engine.make_schedule(groups,cfg)
        self.assertEqual(schedule[0]['learning_rate'],1e-4);self.assertAlmostEqual(schedule[-1]['learning_rate'],1e-5)
        history=[{'round':i+1,'training_samples':5824,'source_round_order_sha256':x['order_sha256'],
          'learning_rate':x['learning_rate'],'optimizer_attempts':728,'optimizer_updates':728} for i,x in enumerate(schedule)]
        self.assertEqual(f.validate_history(history,cfg,schedule)['training_samples'],465920)
        for bad in ['partial','amp','order','lr']:
            h=copy.deepcopy(history)
            if bad=='partial':h.pop()
            elif bad=='amp':h[0]['optimizer_updates']=600
            elif bad=='order':h[0]['source_round_order_sha256']='wrong'
            else:h[0]['learning_rate']=.1
            with self.assertRaises(ValueError):f.validate_history(h,cfg,schedule)

    def test_calibration_rejects_changed_file_and_bad_AMP(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan=[{'fixed':True}];cfg={'training_plan_sha256':f.fingerprint(plan),'code_hashes':{}}
            events=[{'optimizer_update_applied':True,'unscaled_gradient_norm_finite':True,'scale_before':16.,'scale_after':16.}]*32
            history=[{'optimizer_attempts':32,'optimizer_updates':32,'amp_attempt_events':events} for i in range(3)]
            for name,value in [('protocol.json',cfg),('training_plan.json',plan),('history.json',history),('result.json',{'status':'COMPLETED'})]:f.write_json(root/name,value)
            hashes={x.name:f.sha256(x) for x in root.iterdir()}
            with patch.object(f,'FILES17',hashes):
                self.assertEqual(f.calibration(root)[0],cfg)
                history[0]['optimizer_updates']=31;f.write_json(root/'history.json',history)
                with self.assertRaises(ValueError):f.calibration(root)

    def test_bundle_and_backup_exclude_models_and_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'project';code=root/'hycado_start/full_image_training.py';code.parent.mkdir(parents=True);code.write_bytes(b'code')
            out=root/'experiments/18';out.mkdir(parents=True);f.write_json(out/'result.json',{'status':'ERROR'})
            (out/'weights.pt').write_bytes(b'weights');(out/'pixels.png').write_bytes(b'image')
            report=f.bundle(out);f.backup(root,out)
            with zipfile.ZipFile(report) as z:self.assertIsNone(z.testzip());self.assertEqual(z.namelist(),['result.json'])
            self.assertFalse(list((root.parent/'HyCADO_Yedekleri').rglob('weights.pt')))
            checkpoint=out/'weights/epoch_0001.pt';checkpoint.parent.mkdir();checkpoint.write_bytes(b'latest checkpoint')
            f.write_json(out/'checkpoint_index.json',{'latest':{'path':'weights/epoch_0001.pt','sha256':f.sha256(checkpoint)}})
            f.backup(root,out)
            self.assertEqual(next((root.parent/'HyCADO_Yedekleri').rglob('epoch_0001.pt')).read_bytes(),b'latest checkpoint')

    def test_no_GPU_run_never_calls_training_or_changes_old_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'experiments/rtdetr_crop_v1_seed42/keep.pt';old.parent.mkdir(parents=True);old.write_bytes(b'old')
            with patch.object(f,'ROOT',root),patch.object(f,'calibration',return_value=({'upstream_metadata_sha256':{}},[])),\
                patch.object(f.p,'sources',return_value=(root,{},root/'official.pt',{})),\
                patch.object(f,'system_info',return_value={'cuda_available':False}),patch.object(engine,'train') as train,\
                contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertFalse(f.run());train.assert_not_called()
            self.assertEqual(old.read_bytes(),b'old');self.assertEqual(f.read_json(root/'experiments'/f.OUT/'result.json')['status'],'ERROR')


if __name__=='__main__':unittest.main()
