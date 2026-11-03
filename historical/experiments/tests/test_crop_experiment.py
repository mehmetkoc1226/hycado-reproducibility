"""Long-run schedule, committed log recovery and optimizer/EMA replay tests."""
import copy
import hashlib
import json
import math
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from hycado_start import crop_experiment as ce
from hycado_start.stage1 import fingerprint
from hycado_start.common import sha256,write_json

try:
    import torch
    HAVE_TORCH=True
except ImportError:HAVE_TORCH=False


def small_fixture():
    cfg={'seed':42,'rounds':3,'samples_per_round':8,'effective_batch':3,'physical_batch':1,
         'amp':False,'amp_init_scale':16.,'amp_growth_interval':2000,'amp_backoff_factor':.5,'amp_growth_factor':2.,
         'lr':.001,'lrf':.1,'weight_decay':.0001,'betas':[.9,.999],'gradient_clip_norm':10.,
         'ema_decay':.9999,'ema_tau':2000,'minimum_update_fraction_per_round':.90,'imgsz':4}
    groups={'train':[{'source_image':f'train/{i}','crops':[{'source_image':f'train/{i}','crop_index':j,
             'v':(i*3+j+1)/30,'supervision_state':'positive'} for j in range(3)]} for i in range(8)]}
    dev=[{'source_image':'dev/0','crop_index':0,'v':.4,'supervision_state':'positive'}]
    return cfg,groups,dev


class LongScheduleTests(unittest.TestCase):
    def test_cosine_endpoints_monotonic_and_fixed_attempt_budget(self):
        cfg={'lr':1e-4,'lrf':.1,'rounds':80}
        rates=[ce.round_lr(cfg,i) for i in range(80)]
        self.assertEqual(rates[0],1e-4);self.assertAlmostEqual(rates[-1],1e-5)
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
        self.assertEqual(5824*80,465920);self.assertEqual(5824//8*80,58240)
        with self.assertRaises(ValueError):ce.round_lr(cfg,80)

    def test_schedule_one_crop_per_source_cycles_and_is_reproducible(self):
        cfg,groups,dev=small_fixture();schedule=ce.make_schedule(groups,cfg)
        self.assertEqual(schedule,ce.make_schedule(groups,cfg))
        visits={g['source_image']:[] for g in groups['train']}
        for i,row in enumerate(schedule):
            items=ce.source_round(groups['train'],i,cfg['seed'])
            self.assertEqual(len({x['source_image'] for x in items}),8)
            self.assertEqual(ce.order_fingerprint(items),row['order_sha256'])
            for x in items:visits[x['source_image']].append(x['crop_index'])
        self.assertTrue(all(set(v)=={0,1,2} for v in visits.values()))

    def test_journal_discards_only_uncommitted_tail_and_rejects_short_or_corrupt_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events';prefix=b'{"attempt":1}\n';digest=hashlib.sha256(prefix).hexdigest()
            path.write_bytes(prefix+b'partial next round')
            ce.recover_journal(path,len(prefix),digest);self.assertEqual(path.read_bytes(),prefix)
            path.write_bytes(b'short')
            with self.assertRaises(ValueError):ce.recover_journal(path,len(prefix),digest)
            path.write_bytes(b'x'*len(prefix))
            with self.assertRaises(ValueError):ce.recover_journal(path,len(prefix),digest)

    def test_reports_include_small_metadata_and_log_tail_not_model_or_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);write_json(out/'result.json',{'status':'RUNNING'})
            (out/'weights').mkdir();(out/'weights/model.pt').write_bytes(b'large')
            (out/'optimizer_attempts.jsonl').write_text(''.join(json.dumps({'i':i,'padding':'x'*100})+'\n' for i in range(1000)))
            with zipfile.ZipFile(ce.make_bundle(out)) as z:
                self.assertIsNone(z.testzip());self.assertNotIn('weights/model.pt',z.namelist())
                tail=z.read('optimizer_attempts_tail.jsonl').decode()
                self.assertEqual(json.loads(tail.splitlines()[-1])['i'],999)
                for line in tail.splitlines():json.loads(line)
            self.assertFalse(list(out.glob('10_report_*.tmp')))


@unittest.skipUnless(HAVE_TORCH,'Torch required for long loop/checkpoint tests')
class LongEngineTests(unittest.TestCase):
    def _patches(self):
        def model(*args):return torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.Dropout(.2),torch.nn.Linear(5,1))
        def sample(root,item,res,device):return {'img':torch.ones(1,4)*item['v']}
        def loss(model,batch,criterion):
            val=(model(batch['img'])-.3).square().mean();return val,{'loss':val}
        return model,sample,loss

    def test_interrupted_round_replays_lr_ema_scaler_rng_and_exact_journal(self):
        cfg,groups,dev=small_fixture();schedule=ce.make_schedule(groups,cfg);model,sample,loss=self._patches()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=root/'full';b=root/'resume';a.mkdir();b.mkdir()
            with patch.object(ce.calibrated,'build_model',side_effect=model),patch.object(ce,'load_sample',side_effect=sample),\
                 patch('hycado_start.crop_criterion.forward_loss',side_effect=loss):
                ce.train(root,a,None,cfg,groups,schedule,dev,device='cpu')
                calls=0
                def stop(*args):
                    nonlocal calls
                    calls+=1
                    # Complete round 1 (8 train + 1 dev), commit one round-2
                    # optimizer step, then interrupt a partially accumulated window.
                    if calls==14:raise InterruptedError('synthetic partial round')
                    return sample(*args)
                with patch.object(ce,'load_sample',side_effect=stop):
                    with self.assertRaises(InterruptedError):ce.train(root,b,None,cfg,groups,schedule,dev,device='cpu')
                self.assertEqual(ce.read_json(b/'checkpoint_index.json')['latest']['epoch'],1)
                self.assertGreater(len((b/'optimizer_attempts.jsonl').read_text().splitlines()),3)
                ce.train(root,b,None,cfg,groups,schedule,dev,device='cpu')
            def saved(folder):
                return torch.load(folder/ce.read_json(folder/'checkpoint_index.json')['latest']['path'],map_location='cpu',weights_only=False)
            x,y=saved(a),saved(b)
            for key in ['model','ema']:
                for name in x[key]:torch.testing.assert_close(x[key][name],y[key][name],rtol=0,atol=0)
            for i in x['optimizer']['state']:
                for key in x['optimizer']['state'][i]:torch.testing.assert_close(x['optimizer']['state'][i][key],y['optimizer']['state'][i][key],rtol=0,atol=0)
            self.assertEqual(x['scaler'],y['scaler']);self.assertEqual(x['ema_updates'],y['ema_updates'])
            torch.testing.assert_close(x['rng']['torch'],y['rng']['torch'],rtol=0,atol=0)
            self.assertEqual((a/'optimizer_attempts.jsonl').read_bytes(),(b/'optimizer_attempts.jsonl').read_bytes())
            self.assertEqual([h['learning_rate'] for h in y['history']],[ce.round_lr(cfg,i) for i in range(3)])
            self.assertEqual(y['journal_offset'],(b/'optimizer_attempts.jsonl').stat().st_size)
            # Completed runs rebuild sidecars without constructing/training a model.
            (b/'history.json').unlink()
            with patch.object(ce.calibrated,'build_model',side_effect=AssertionError('must not retrain')):
                history=ce.train(root,b,None,cfg,groups,schedule,dev,device='cpu')
            self.assertEqual(len(history),3)

    def test_failed_checkpoint_leaves_previous_round_recoverable(self):
        cfg,groups,dev=small_fixture();schedule=ce.make_schedule(groups,cfg);model,sample,loss=self._patches()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'out';out.mkdir();real=ce.commit_checkpoint
            def fail(directory,epoch,writer,digest):
                if epoch==2:raise OSError('synthetic disk failure')
                return real(directory,epoch,writer,digest)
            with patch.object(ce.calibrated,'build_model',side_effect=model),patch.object(ce,'load_sample',side_effect=sample),\
                 patch('hycado_start.crop_criterion.forward_loss',side_effect=loss):
                with patch.object(ce,'commit_checkpoint',side_effect=fail):
                    with self.assertRaises(OSError):ce.train(root,out,None,cfg,groups,schedule,dev,device='cpu')
                self.assertEqual(ce.read_json(out/'checkpoint_index.json')['latest']['epoch'],1)
                self.assertEqual(len(ce.read_json(out/'history.json')),1)
                history=ce.train(root,out,None,cfg,groups,schedule,dev,device='cpu')
            self.assertEqual(len(history),3)

    def test_corrupted_order_or_optimizer_state_is_rejected(self):
        cfg,groups,dev=small_fixture();schedule=ce.make_schedule(groups,cfg);model,sample,loss=self._patches()
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);changed=copy.deepcopy(schedule);changed[0]['order_sha256']='bad'
            with patch.object(ce.calibrated,'build_model',side_effect=model):
                with self.assertRaisesRegex(ValueError,'sirasi'):ce.train(out,out,None,cfg,groups,changed,dev,device='cpu')
        tiny=model();opt=torch.optim.AdamW(tiny.parameters());ema=type('E',(),{'ema':copy.deepcopy(tiny)})()
        next(tiny.parameters()).data.fill_(float('nan'))
        with self.assertRaises(FloatingPointError):ce.finite_states(tiny,ema,opt)


if __name__=='__main__':unittest.main()
