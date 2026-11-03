"""AMP scale changes, rejection rules and recovery for the isolated v2 runner."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from hycado_start import crop_training_v2 as v2
    import test_crop_training as original_tests
    HAVE_TORCH=True
except ImportError:HAVE_TORCH=False


@unittest.skipUnless(HAVE_TORCH,'Torch required for AMP tests')
class AMPRevisionTests(unittest.TestCase):
    def test_real_grad_scaler_finite_step_keeps_scale_and_applies_update(self):
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1.)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=16.,growth_interval=2000)
        scaler.scale(model(torch.ones(1,1)).square().sum()).backward()
        record=v2.scaled_optimizer_step(model,opt,scaler,10.)
        self.assertTrue(record['optimizer_update_applied']);self.assertTrue(record['unscaled_gradient_norm_finite'])
        self.assertEqual(record['scale_before'],16.);self.assertEqual(record['scale_after'],16.)
        self.assertAlmostEqual(record['unscaled_gradient_norm'],2.)
        self.assertAlmostEqual(float(model.weight.detach()),.8,places=6)

    def test_real_scaler_infinite_gradient_skips_and_halves_then_recovers(self):
        model=torch.nn.Linear(1,1,bias=False);model.weight.data.fill_(1.)
        opt=torch.optim.SGD(model.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cpu',init_scale=16.,growth_interval=2000)
        scaler.scale(model(torch.ones(1,1)).sum()*torch.tensor(float('inf'))).backward()
        record=v2.scaled_optimizer_step(model,opt,scaler,10.)
        self.assertFalse(record['optimizer_update_applied']);self.assertEqual(float(model.weight.detach()),1.)
        self.assertEqual(record['scale_after'],8.);self.assertIsNone(record['unscaled_gradient_norm'])
        json.dumps(record,allow_nan=False)
        # Actual scaler state survives a round trip and the next finite update.
        replacement=torch.amp.GradScaler('cpu',init_scale=16.,growth_interval=2000)
        replacement.load_state_dict(scaler.state_dict());self.assertEqual(replacement.get_scale(),8.)
        opt.zero_grad(set_to_none=True)
        replacement.scale(model(torch.ones(1,1)).square().sum()).backward()
        recovered=v2.scaled_optimizer_step(model,opt,replacement,10.)
        self.assertTrue(recovered['optimizer_update_applied']);self.assertEqual(recovered['scale_before'],8.)

    def test_cuda_scaler_factory_receives_explicit_parameters(self):
        cfg={'amp':True,'amp_init_scale':16.,'amp_growth_interval':2000,
             'amp_growth_factor':2.,'amp_backoff_factor':.5}
        with patch('torch.amp.GradScaler') as make:
            v2.make_scaler(cfg,'cuda')
        make.assert_called_once_with('cuda',init_scale=16.,growth_factor=2.,backoff_factor=.5,growth_interval=2000,enabled=True)

    def test_v1_failure_is_still_rejected_by_quality_gate(self):
        history=[{'phase':i+1,'optimizer_attempts':32,'optimizer_updates':n} for i,n in enumerate([20,32,32])]
        with self.assertRaises(RuntimeError):v2.validate_amp_history(history)

    def test_global_pass_does_not_hide_a_bad_phase(self):
        history=[{'phase':i+1,'optimizer_attempts':32,'optimizer_updates':n} for i,n in enumerate([32,32,24])]
        self.assertGreater(sum(x['optimizer_updates'] for x in history)/96,.90)
        with self.assertRaisesRegex(RuntimeError,'blok 3'):v2.validate_amp_history(history)
        for h in history:h['optimizer_updates']=32
        self.assertEqual(v2.validate_amp_history(history)['overall_update_fraction'],1.)

    def test_reference_rejects_changed_data_weights_upstream_or_plan(self):
        plan=[{'phase':1}]
        cfg={'catalog':{'a':1},'initial_weights_sha256':'weights','upstream_hashes':{'file':'hash'}}
        cfg['calibration_reference']={**copy.deepcopy(cfg),'training_plan_sha256':v2.fingerprint(plan)}
        v2.validate_reference(cfg,plan)
        for key in ['catalog','initial_weights_sha256','upstream_hashes']:
            changed=copy.deepcopy(cfg);changed[key]='changed'
            with self.assertRaises(ValueError):v2.validate_reference(changed,plan)
        with self.assertRaises(ValueError):v2.validate_reference(cfg,[{'phase':2}])

    def test_v2_recovery_matches_uninterrupted_optimizer_ema_and_rng(self):
        # Reuse the substantive dropout/AdamW/EMA interruption scenario against
        # the new loop; the original v1 implementation is not patched on disk.
        with patch('test_crop_training.ct',v2):
            original_tests.EngineRecoveryTests.test_interrupted_phase_resume_matches_uninterrupted_optimizer_ema_and_rng(self)

    def test_cpu_launch_reports_error_and_preserves_v1_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'experiments/rtdetr_crop_calibration_v1_seed42';old.mkdir(parents=True)
            marker=old/'keep.txt';marker.write_bytes(b'preserve old run')
            with patch.object(v2,'ROOT',root),patch('torch.cuda.is_available',return_value=False):
                self.assertFalse(v2.run())
            self.assertEqual(marker.read_bytes(),b'preserve old run')
            new=root/'experiments/rtdetr_crop_calibration_v2_seed42'
            with zipfile.ZipFile(new/'09_rtdetr_crop_v2_rapor.zip') as z:
                self.assertEqual(json.loads(z.read('result.json'))['status'],'ERROR')
                self.assertNotIn('budget_projection.json',z.namelist())


if __name__=='__main__':unittest.main()
