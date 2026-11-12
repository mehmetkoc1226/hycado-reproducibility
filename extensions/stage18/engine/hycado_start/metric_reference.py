"""24: export immutable frozen predictions for independent reference execution.

Default Windows operation only prepares inputs. It never certifies parity,
loads a model, imports torch, or changes stages 01-23. The separate reference
runner executes the byte-identical pinned .m functions under GNU Octave.
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, os, shutil, subprocess, sys
import tempfile, time, traceback, zipfile
from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat
from .common import ROOT, sha256, utc_now, write_json
from .stage1 import fingerprint, training_lock
from .dev_metrics import raw_annotation, ignore_integral, prepare_groundtruth, outside_ignore, xyxy_to_det

OUT = 'metric_reference_v1_seed42'
NAMES = ['baseline','rtdetr_full960','fixed_grid_a025','hybrid_a025',
         'relocated_s0_a025','relocated_s1_a025','relocated_s2_a025']
KEYS = ['AP','AP50','AP75','AR1','AR10','AR100','AR500']
TOLERANCE = 1e-7  # percentage points; frozen before reference execution
REPORT = '24_metrik_dogrulama_rapor.zip'


def reference_assets(root):
    folder = Path(root)/'reference24'
    manifest = json.loads((folder/'reference_manifest.json').read_text(encoding='utf-8'))
    for name, entry in manifest['files'].items():
        path = folder/'toolkit'/(name+'.m')
        if sha256(path) != entry['sha256']:
            raise ValueError('Sabit referans kaynak hash farki: '+name)
        blob = b'blob '+str(path.stat().st_size).encode()+b'\0'+path.read_bytes()
        if hashlib.sha1(blob).hexdigest() != entry['git_blob_sha']:
            raise ValueError('Sabit Git blob kimligi farkli: '+name)
    return folder, manifest


def snapshot_assets(source, output):
    target = output/'reference24'
    for path in sorted(source.rglob('*')):
        if not path.is_file(): continue
        dest = target/path.relative_to(source)
        if dest.exists() and sha256(dest) != sha256(path):
            raise ValueError('24 onceki referans dosyasi degismis: '+str(dest))
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,dest)
    return target


def source(root, manifest):
    from . import paired_bootstrap as previous
    c12, ctx, immutable = previous.source(root)
    folder = Path(root)/'experiments'/previous.OUT
    for name, digest in manifest['completed23_files'].items():
        if sha256(folder/name) != digest:
            raise ValueError('Tamamlanan 23 kaydi farkli: '+name)
        immutable[str(folder/name)] = digest
    cfg = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    result = json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if (cfg['records'] != c12['records'] or cfg['names'] != NAMES or
        result.get('status') != 'COMPLETED' or result.get('backup_verified') is not True):
        raise ValueError('Tamamlanan ve yedegi dogrulanan 23 gerekli')
    for name, digest in cfg['code_sha256'].items():
        if sha256(Path(previous.__file__).with_name(name)) != digest:
            raise ValueError('23 kaynak kodu degismis: '+name)
    index = json.loads((folder/'cache_index.json').read_text(encoding='utf-8'))['records']
    if [x['source_image'] for x in index] != [x['image'] for x in c12['records']]:
        raise ValueError('23 cache sirasi farkli')
    metrics = json.loads((folder/'point_metrics.json').read_text(encoding='utf-8'))['variants']
    return c12['records'], ctx, cfg, folder, index, metrics, immutable


def det8(rows):
    a = xyxy_to_det(rows)
    return np.column_stack([a, np.full((len(a),2),-1.,dtype=np.float64)])


def make_input(raw_text, rows, part, width, height):
    """Preserve all final rows, including rows below a later max500 cutoff."""
    raw = raw_annotation(raw_text)
    integral = ignore_integral(raw,width,height)
    gt = prepare_groundtruth(raw,integral)
    counts = np.array([np.count_nonzero(gt[:,5] == c) for c in range(1,11)],dtype=np.int64)
    if not np.array_equal(counts,part['gt_denominators']):
        raise ValueError('24 GT sayaci sabit 23 ile farkli')
    data = {'rawgt':raw,'sizes':np.array([[height,width]],dtype=np.int64),
            'python_gt':gt, 'python_denominators':counts.reshape(1,-1),
            'python_thresholds':np.linspace(.5,.95,10).reshape(1,-1)}
    for i,name in enumerate(NAMES):
        a = det8(rows[name]); prepared = outside_ignore(a,integral); short = prepared[:500]
        if (not np.array_equal(short[:,4],part[name+'_scores']) or
            not np.array_equal(short[:,5],part[name+'_classes'])):
            raise ValueError('24 skor/sinif sabit 23 ile farkli: '+name)
        data['raw'+str(i)] = a
        data['python_det'+str(i)] = prepared
        data['python_flags'+str(i)] = part[name+'_flags']
    return data


def commit_input(folder, data, annotation, bindings):
    """Atomic per-image commit; a restart never reuses an incomplete image."""
    folder = Path(folder); staging = folder.parent.parent/'staging'
    staging.mkdir(parents=True,exist_ok=True)
    temp = Path(tempfile.mkdtemp(dir=staging,prefix=folder.name+'_'))
    try:
        data = dict(data); data['binding_token'] = fingerprint(bindings)
        savemat(temp/'input.mat',data,do_compression=True,oned_as='row')
        (temp/'annotation.txt').write_bytes(annotation)
        meta = {'bindings':bindings,'input_sha256':sha256(temp/'input.mat'),
                'annotation_sha256':sha256(temp/'annotation.txt')}
        meta['content_sha256'] = fingerprint(meta)
        write_json(temp/'bindings.json',meta)
        if folder.exists(): raise FileExistsError('24 tamamlanmis girdinin uzerine yazilmadi')
        folder.parent.mkdir(parents=True,exist_ok=True); temp.rename(folder)
    finally:
        if temp.exists(): shutil.rmtree(temp)


def checked_input(folder, bindings):
    folder = Path(folder)
    meta = json.loads((folder/'bindings.json').read_text(encoding='utf-8'))
    if (meta['bindings'] != bindings or meta['content_sha256'] != fingerprint({k:v for k,v in meta.items() if k!='content_sha256'}) or
        sha256(folder/'input.mat') != meta['input_sha256'] or
        sha256(folder/'annotation.txt') != meta['annotation_sha256'] or
        meta['annotation_sha256'] != bindings['annotation_sha256']):
        raise ValueError('24 yeniden kullanilacak girdi degismis: '+str(folder))
    return meta


def export_images(root, output, cfg, ctx, cfg23, folder23, index23, immutable):
    from . import paired_bootstrap as previous
    signature = fingerprint(cfg); entries = []; fresh = 0; start = time.perf_counter()
    for i,(record,cache) in enumerate(zip(cfg['records'],index23)):
        ann = Path(root)/'data/prepared/visdrone_annotations/val'/(Path(record['image']).stem+'.txt')
        if sha256(ann) != record['annotation_sha256']: raise ValueError('24 ham annotation degismis')
        immutable[str(ann)] = record['annotation_sha256']
        paths = previous.paths_for(record,ctx)
        for path,digest in paths.values():
            if sha256(path) != digest: raise ValueError('24 kaynak tahmin degismis')
            immutable[str(path)] = digest
        old = previous.safe_path(folder23,cache['path'])
        if sha256(old/'summary.json') != cache['summary_sha256'] or sha256(old/'matches.npz') != cache['npz_sha256']:
            raise ValueError('23 eslesme cache degismis')
        immutable[str(old/'summary.json')] = cache['summary_sha256']; immutable[str(old/'matches.npz')] = cache['npz_sha256']
        old_bindings = {'protocol_sha256':fingerprint(cfg23),'source_image':record['image'],
            'annotation_sha256':record['annotation_sha256'],'source_prediction_sha256':{k:d for k,(_,d) in paths.items()}}
        bindings = {'protocol_sha256':signature,'source_image':record['image'],
            'annotation_sha256':record['annotation_sha256'],'source_prediction_sha256':old_bindings['source_prediction_sha256'],
            'cache23_sha256':cache['npz_sha256'],'cache23_summary_sha256':cache['summary_sha256']}
        dest = Path(output)/'items'/f'{i:04d}'
        if not dest.exists():
            part,_ = previous.read_part(old,old_bindings)
            data = make_input(ann.read_text(encoding='utf-8-sig'),previous.load_rows(record,ctx),part,record['width'],record['height'])
            commit_input(dest,data,ann.read_bytes(),bindings); fresh += 1
        meta = checked_input(dest,bindings)
        entries.append({'id':i,'source_image':record['image'],'path':dest.relative_to(output).as_posix(),
            'bindings_sha256':sha256(dest/'bindings.json'),'input_sha256':meta['input_sha256'],
            'annotation_sha256':meta['annotation_sha256'],'binding_token':fingerprint(bindings)})
        if i==0 or (i+1)%25==0 or i+1==len(cfg['records']):
            elapsed = time.perf_counter()-start
            print(f'24 girdi hazirligi: {i+1}/{len(cfg["records"])}; yeni={fresh}; gecen={elapsed/60:.1f} dk',flush=True)
            write_json(Path(output)/'progress.json',{'status':'EXPORTING','images_done':i+1,'new_images':fresh})
    write_json(Path(output)/'input_index.json',{'protocol_sha256':signature,'records':entries})
    return entries


def bundle(output):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    fd,temp = tempfile.mkstemp(dir=output,suffix='.zip.tmp'); os.close(fd)
    target = output/REPORT
    try:
        with zipfile.ZipFile(temp,'w') as z:
            for path in sorted(output.rglob('*')):
                rel = path.relative_to(output)
                if path.is_file() and 'staging' not in rel.parts and path.suffix in ['.json','.mat','.txt','.m','.md','.py']:
                    z.write(path,rel.as_posix(),compress_type=zipfile.ZIP_STORED if path.suffix=='.mat' else zipfile.ZIP_DEFLATED)
            log = output/'console.log'
            if log.exists(): z.writestr('console_tail.log',log.read_bytes()[-90000:],compress_type=zipfile.ZIP_DEFLATED)
        os.replace(temp,target)
    finally: Path(temp).unlink(missing_ok=True)
    return target


def export(root=ROOT, *, expected_images=548):
    from .__main__ import _Tee
    root = Path(root); output = root/'experiments'/OUT; output.mkdir(parents=True,exist_ok=True)
    status = 'ERROR'
    with training_lock(output/'.metric24.lock'):
        with (output/'console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(_Tee(sys.stdout,log)),contextlib.redirect_stderr(_Tee(sys.stderr,log)):
                try:
                    print('24: REFERANS METRIK GIRDILERI HAZIRLANIYOR.',utc_now(),flush=True)
                    print('Egitim/GPU/model yukleme yok. Sonuc READY_FOR_REFERENCE olacak; henuz parity degil.',flush=True)
                    write_json(output/'result.json',{'status':'EXPORTING','reference_executed':False})
                    assets,manifest = reference_assets(root)
                    records,ctx,c23,f23,index,metrics,immutable = source(root,manifest)
                    if len(records)!=expected_images: raise ValueError('Sabit VAL goruntu sayisi farkli')
                    cfg = {'id':'hycado_reference_inputs_v1','date':'2026-11-03','records':records,'names':NAMES,'keys':KEYS,
                        'reference_commit':manifest['commit'],'tolerance_percentage_points':TOLERANCE,
                        'matching_acceptance':'zero per-image GT/detection preprocessing and match-flag differences',
                        'reference_execution':'separate GNU Octave execution; never certify at export; no MATLAB binary claim',
                        'files23':manifest['completed23_files'],'code_sha256':sha256(__file__),
                        'assets_sha256':{p.relative_to(assets).as_posix():sha256(p) for p in sorted(assets.rglob('*')) if p.is_file()},
                        'all_final_predictions_retained_before_ignore_and_max500':True,'training_started':False,'GPU_inference':False}
                    cfg['python_metrics_fingerprint'] = fingerprint({'variants':metrics})
                    if (output/'protocol.json').exists() and json.loads((output/'protocol.json').read_text(encoding='utf-8'))!=cfg:
                        raise ValueError('24 onceki protokol/kod farkli; eski girdiyle karistirilmadi')
                    write_json(output/'protocol.json',cfg); write_json(output/'python_metrics.json',{'variants':metrics})
                    write_json(output/'environment.json',{'created_utc':utc_now(),'python':sys.version,'numpy':np.__version__,
                        'scipy':__import__('scipy').__version__,'training_started':False,'GPU_inference':False})
                    snapshot_assets(assets,output); shutil.copyfile(__file__,output/'metric_reference_snapshot.py')
                    if (output/'staging').exists(): shutil.rmtree(output/'staging')
                    entries = export_images(root,output,cfg,ctx,c23,f23,index,immutable)
                    for path,digest in immutable.items():
                        if sha256(path)!=digest: raise ValueError('24 islem sirasinda kaynak degismis: '+path)
                    write_json(output/'source_integrity.json',{'files_checked':len(immutable),'all_unchanged':True,'files':immutable})
                    status = 'READY_FOR_REFERENCE'
                    result = {'status':status,'images':len(entries),'variants':len(NAMES),'reference_executed':False,
                        'pinned_reference_source_parity_verified':False,'official_binary_parity_verified':False,
                        'training_started':False,'GPU_inference':False,'original_results_unchanged':True}
                    (output/'READ_ME.md').write_text('24 inputs ready. This is NOT a parity result. Send the complete 24 ZIP for independent reference execution.\n',encoding='utf-8')
                except (Exception,KeyboardInterrupt) as exc:
                    traceback.print_exc();status = 'INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'ERROR'
                    result = {'status':status,'error':str(exc),'reference_executed':False,'resume':'rerun 24; completed images retained'}
                write_json(output/'result.json',result); print('SONUC:',status,flush=True)
        report = bundle(output); print('GONDERILECEK DOSYA:',report,flush=True)
    return status=='READY_FOR_REFERENCE'


def compare_output(input_path, output_path, token):
    """Read-only diagnostic comparison of genuine reference component results."""
    a = loadmat(input_path); b = loadmat(output_path); errors = []
    actual_token = str(np.asarray(b['binding_token']).reshape(-1)[0])
    if actual_token != token: raise ValueError('Referans cikti girdi kimligi farkli')
    for name in ['gt']+['det'+str(i) for i in range(len(NAMES))]:
        x = np.asarray(a['python_'+name]); y = np.asarray(b[name])
        if x.shape!=y.shape or not np.array_equal(x,y): errors.append(name)
    for i,name in enumerate(NAMES):
        x = np.asarray(a['python_flags'+str(i)]); y = np.asarray(b['flags'+str(i)])
        if x.shape!=y.shape or not np.array_equal(x,y): errors.append('flags:'+name)
    return errors


def execute_worker(executable, assets, mode, input_dir, output_dir, variant=-1):
    def mq(value): return "'"+str(value).replace("'","''")+"'"
    expression = 'addpath('+mq(assets)+');hycado_reference24('+','.join([mq(mode),mq(input_dir),mq(output_dir),str(variant)])+');'
    cmd = [str(executable),'--no-gui','--quiet','--eval',expression]
    output_dir = Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/f'{mode}_{variant}_console.log').open('w',encoding='utf-8') as log:
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)


def run_reference(input_dir, executable):
    """Separate host-side runner; no arbitrary ZIP extraction or model imports."""
    inp = Path(input_dir).resolve(); assets,manifest = reference_assets(inp)
    cfg = json.loads((inp/'protocol.json').read_text(encoding='utf-8'))
    index = json.loads((inp/'input_index.json').read_text(encoding='utf-8'))
    ready = json.loads((inp/'result.json').read_text(encoding='utf-8'))
    if ready.get('status')!='READY_FOR_REFERENCE' or ready.get('reference_executed') is not False:
        raise ValueError('24 eksiksiz girdi hazirligi gerekli; mevcut dosya bir parity raporu degil')
    if len(index['records'])!=len(cfg['records']) or not cfg['records']:
        raise ValueError('24 eksik veya bos girdi dizini')
    if cfg['names'] != NAMES or index['protocol_sha256'] != fingerprint(cfg): raise ValueError('24 girdi protokolu farkli')
    python_metrics = json.loads((inp/'python_metrics.json').read_text(encoding='utf-8'))
    if fingerprint(python_metrics)!=cfg['python_metrics_fingerprint']:
        raise ValueError('24 sabit Python karsilastirma metrikleri degismis')
    for name,digest in cfg['assets_sha256'].items():
        if sha256(assets/name)!=digest: raise ValueError('24 referans varligi farkli: '+name)
    for rec in index['records']:
        folder = (inp/rec['path']).resolve()
        if not folder.is_relative_to(inp): raise ValueError('24 girdi yolu klasor disinda')
        for name,key in [('input.mat','input_sha256'),('annotation.txt','annotation_sha256'),('bindings.json','bindings_sha256')]:
            if sha256(folder/name)!=rec[key]: raise ValueError('24 referans girdisi degismis')
    output = inp/'reference_run'; output.mkdir(exist_ok=True)
    job = {'protocol_sha256':fingerprint(cfg),'inputs_sha256':sha256(inp/'input_index.json'),
           'executable':str(Path(executable).resolve()),'executable_sha256':sha256(executable),'runner_code_sha256':sha256(__file__)}
    if (output/'job.json').exists() and json.loads((output/'job.json').read_text())!=job:
        raise ValueError('24 onceki referans isi farkli')
    write_json(output/'job.json',job)
    immutable = {str(p):sha256(p) for p in inp.rglob('*') if p.is_file() and not p.is_relative_to(output)}
    execute_worker(executable,assets,'components',inp,output)
    components = []
    for rec in index['records']:
        result_path = output/'components'/f'{rec["id"]:04d}.mat'
        errors = compare_output(inp/rec['path']/'input.mat',result_path,rec['binding_token'])
        components.append({'image':rec['source_image'],'differences':errors,'reference_output_sha256':sha256(result_path)})
    write_json(output/'component_comparison.json',components)
    # All seven unchanged calcAccuracy runs are required even if components disagree.
    expected = python_metrics['variants']; measured = {}; passed = not any(x['differences'] for x in components)
    for i,name in enumerate(NAMES):
        execute_worker(executable,assets,'literal',inp,output,i)
        result = loadmat(output/f'literal_{i}.mat')
        if str(np.asarray(result['binding_token']).reshape(-1)[0]) != fingerprint(cfg): raise ValueError('Literal referans protokolu farkli')
        values = np.asarray(result['values'],dtype=np.float64).reshape(-1)
        target = np.array([expected[name][k] for k in KEYS])
        delta = np.abs(values-target); ok = bool(np.isfinite(delta).all() and np.all(delta<=TOLERANCE))
        measured[name] = {'reference_values':dict(zip(KEYS,values.tolist())),
            'python_values':dict(zip(KEYS,target.tolist())),'absolute_differences':dict(zip(KEYS,delta.tolist())),'passed':ok}
        passed &= ok
        write_json(output/'metric_comparison.json',measured)
    for path,digest in immutable.items():
        if sha256(path)!=digest: raise ValueError('Referans calisirken girdi degismis: '+path)
    result = {'status':'PASSED' if passed else 'FAILED','images':len(index['records']),'variants':len(NAMES),
        'reference_executed':True,'pinned_reference_source_parity_verified':passed,
        'official_binary_parity_verified':False,'MATLAB_execution_performed':False,
        'interpreter':'GNU Octave; exact version stored in component MAT files',
        'official_leaderboard_score_certified':False,'tolerance_percentage_points':TOLERANCE,
        'reference_commit':manifest['commit'],'components_with_differences':sum(bool(x['differences']) for x in components),
        'source_files_rechecked':len(immutable),
        'original_results_unchanged':True,'training_started':False,'GPU_inference':False}
    write_json(output/'result.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--report-only',action='store_true')
    parser.add_argument('--reference-inputs'); parser.add_argument('--octave')
    args = parser.parse_args()
    if args.report_only: print(bundle(ROOT/'experiments'/OUT)); return 0
    if args.reference_inputs:
        if not args.octave: parser.error('--octave executable gerekli')
        return 0 if run_reference(args.reference_inputs,args.octave)['status']=='PASSED' else 1
    return 0 if export() else 1


if __name__=='__main__': raise SystemExit(main())
