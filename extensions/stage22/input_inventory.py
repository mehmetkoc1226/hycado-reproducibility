"""Read-only HyCADO input inventory. Standard library only; no scoring or model imports."""
from __future__ import annotations
import argparse, ast, collections, csv, hashlib, io, json, math, os
from pathlib import Path, PurePosixPath
import sqlite3, struct, sys, traceback, zipfile
from datetime import datetime, timezone
from contextlib import closing

HERE = Path(__file__).resolve().parent
RUNS = {18: 'testdev_frozen_v1_seed42', 20: 'testdev1280_extension_v1_seed42'}
ISSUES = []

def require(ok, text):
    if not ok:
        raise ValueError(text)

def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def stat(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns)

def safe(base, relative):
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and '\\' not in relative and
            all(part not in ('..', '.') for part in rel.parts), 'Invalid relative path')
    p = base.joinpath(*rel.parts).resolve()
    require(p.is_relative_to(base.resolve()), 'Path escapes source directory')
    return p

def samples(records):
    return [records[i] for i in sorted({0, len(records)//2, len(records)-1})] if records else []

def readonly(path):
    # Opening a missing SQLite path must never create a database.
    path = Path(path).resolve()
    require(path.is_file(), 'Database missing: ' + str(path))
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True, timeout=10)
    db.execute('PRAGMA query_only=ON')
    return db

def validate_rows(body, rec, *, max_candidates=1000, expected_count=None, label='prediction'):
    require(body['image_sha256'] == rec['image_sha256'], label+': prediction image identity differs')
    rows = body['detections']
    require(isinstance(rows, list), label+': final candidate rows must be a list')
    if expected_count is not None:
        require(type(expected_count) is int and expected_count >= 0,
                label+': invalid frozen candidate count')
        require(len(rows) == expected_count,
                label+': candidate count differs from frozen index; expected '+str(expected_count)+', got '+str(len(rows)))
    if max_candidates is not None:
        require(len(rows) <= max_candidates,
                label+': source method candidate cap exceeded; cap '+str(max_candidates)+', got '+str(len(rows)))
    for row in rows:
        require(len(row) == 6 and all(isinstance(x, (int,float)) and math.isfinite(x) for x in row),
                label+': prediction contains invalid numeric values')
        x1,y1,x2,y2,score,cls = row
        require(0 <= x1 < x2 <= rec['width'] and 0 <= y1 < y2 <= rec['height'] and
                0 <= score <= 1 and cls == int(cls) and 0 <= cls <= 9, label+': prediction geometry/class differs')
    return len(rows)

def validate_p2_rows(body, rec, method, entry):
    # P2's source runner caps individual YOLO outputs at 1000, but does not
    # globally cap SAHI's merged archive. Never truncate frozen SAHI rows.
    require(method in ('yolo800_repeat','yolo1280','yolo1536','sahi640','sahi800'),
            'Unknown P2 source method: '+str(method))
    require(type(entry.get('detections')) is int and entry['detections'] >= 0,
            'Frozen P2 index lacks a valid candidate count: '+method)
    label=method+'/'+entry['source_image']
    return validate_rows(body,rec,max_candidates=None if method in ('sahi640','sahi800') else 1000,
                         expected_count=entry['detections'],label=label)

def npy_header(data):
    require(data[:6] == b'\x93NUMPY', 'Invalid NPY header')
    major = data[6]
    require(major in (1,2,3), 'Unknown NPY version')
    n = 2 if major == 1 else 4
    size = int.from_bytes(data[8:8+n], 'little')
    require(size < 16384, 'Excessive NPY header')
    d = ast.literal_eval(data[8+n:8+n+size].decode('utf-8' if major == 3 else 'latin1').strip())
    return {'shape': list(d['shape']), 'dtype': d['descr']}

def database_check(path, protocol_sha, records, index, methods, deep=False, raw=False):
    before = stat(path)
    report = {'path': str(path), 'bytes': before[0], 'full_database_hash_checked': False,
              'sampled_images_per_method': len(samples(records))}
    by_stem = {Path(x['image']).stem:x for x in records}
    require(len(by_stem) == len(records), 'Duplicate source stems')
    idx = {Path(x['image']).stem:x for x in index['records']}
    require(set(idx) == set(by_stem), 'Index/source coverage differs')
    with closing(readonly(path)) as db:
        require(db.execute('SELECT signature FROM binding WHERE id=1').fetchall() == [(protocol_sha,)],
                'SQLite protocol binding differs')
        actual = dict(db.execute('SELECT method,COUNT(*) FROM prediction GROUP BY method'))
        require(actual == {m:len(records) for m in methods}, 'SQLite method/row counts differ: '+str(actual))
        # All stems checked, but only deterministic payload samples decompressed at this step.
        for method in methods:
            require({x[0] for x in db.execute('SELECT stem FROM prediction WHERE method=?',(method,))} == set(by_stem),
                    'SQLite stem coverage differs')
        checked = {}
        import zlib
        for method in methods:
            ns = []
            for rec in samples(records):
                stem = Path(rec['image']).stem
                row = db.execute('SELECT digest,payload FROM prediction WHERE method=? AND stem=?',
                                 (method,stem)).fetchone()
                plain = zlib.decompress(row[1])
                require(digest_bytes(plain) == row[0] == idx[stem]['variant_sha256'][method],
                        'Sample prediction digest differs')
                ns.append(validate_rows(json.loads(plain), rec))
            checked[method] = ns
        report.update({'methods':actual, 'sample_detection_counts':checked, 'all_stems_verified':True})
        if raw:
            require(db.execute('SELECT COUNT(*) FROM crop_raw').fetchone()[0] == len(records),
                    'Raw crop-query coverage differs')
            require({x[0] for x in db.execute('SELECT stem FROM crop_raw')} == set(by_stem),
                    'Raw crop stems differ')
            descriptions = []
            for rec in samples(records):
                stem = Path(rec['image']).stem
                row = db.execute('SELECT digest,payload FROM crop_raw WHERE stem=?',(stem,)).fetchone()
                require(digest_bytes(row[1]) == row[0] == idx[stem]['raw_sha256'], 'Raw crop sample digest differs')
                with zipfile.ZipFile(io.BytesIO(row[1])) as z:
                    require(set(z.namelist()) == {'queries.npy','rects.npy'}, 'Raw crop archive members differ')
                    q, rect = npy_header(z.read('queries.npy')), npy_header(z.read('rects.npy'))
                    require(len(q['shape']) == 3 and q['shape'][1:] == [300,14] and
                            q['dtype'] in ('<f4','=f4') and rect['shape'] == [q['shape'][0],4],
                            'Raw crop shape differs')
                descriptions.append({'image':rec['image'],'queries':q,'rects':rect})
            report['raw_crop_samples'] = descriptions
    if deep:
        report.update({'sha256':digest(path), 'full_database_hash_checked':True})
    require(stat(path) == before, 'Database changed during inventory; rerun after active writers stop')
    report['source_stat_unchanged'] = True
    return report

def annotation_check(folder, records, expected):
    counts = collections.Counter()
    examples = []
    for i, rec in enumerate(records,1):
        p = folder / rec['annotation']
        before = stat(p)
        data = p.read_bytes()
        require(digest_bytes(data) == expected[rec['stem']], 'Annotation SHA differs: '+p.name)
        for number, fields in enumerate(csv.reader(data.decode('utf-8-sig').splitlines()),1):
            while fields and not fields[-1].strip():
                fields.pop()
            if not fields:
                continue
            require(len(fields) == 8, 'Annotation must have 8 columns: '+p.name)
            row = [float(x) for x in fields]
            require(all(math.isfinite(x) for x in row) and row[4] in (0,1) and
                    row[5] == int(row[5]) and 0 <= row[5] <= 11, 'Invalid annotation: '+p.name)
            cls,score = int(row[5]),int(row[4])
            counts[f'class_{cls}_score_{score}'] += 1
            if row[2] <= 0 or row[3] <= 0:
                counts['nonpositive_geometry'] += 1
                if len(examples) < 10:examples.append({'file':p.name,'line':number,'reason':'nonpositive_geometry'})
            if 1 <= cls <= 10 and score == 0:
                counts['class_specific_ignored'] += 1
            if 1 <= cls <= 10 and score == 1:
                counts['normal_targets'] += 1
        require(stat(p) == before, 'Annotation changed during inventory')
        if i in (1,len(records)) or i%500 == 0:
            print(f'Annotation inventory: {i}/{len(records)}',flush=True)
    return {'files_sha256_verified':len(records), 'counts':dict(counts), 'geometry_examples':examples}

def checked_reference(stage, project):
    ref = HERE/'references'/('stage'+str(stage))
    folder = project/'experiments'/RUNS[stage]
    for name in ('protocol.json','result.json','prediction_index.json'):
        require(digest(folder/name) == digest(ref/name), f'Stage {stage} report bytes differ: {name}')
    protocol = read(ref/'protocol.json')
    sig = digest_bytes(json.dumps(protocol,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode())
    result = read(ref/'result.json')
    require(result['protocol_sha256'] == sig and result['status'] == 'COMPLETED', 'Protocol signature differs')
    return folder, protocol, result, read(ref/'prediction_index.json'), sig

def locate_dataset(old, manifest, override):
    if override:
        target = Path(override).resolve()
    elif manifest['official_zip_sha256']:
        target = old/('dataset_'+manifest['official_zip_sha256'][:16])
    else:
        raise ValueError('Original test-dev location requires --dataset')
    require((target/'images').is_dir() and (target/'annotations').is_dir(),
            'Test-dev dataset missing; provide --dataset with images/ and annotations/')
    return target

def val_check(project):
    ref = HERE/'references/p2_val'
    folder = project/'experiments/p2_strong_baselines_v1_seed42'
    for name in ('official_val_protocol.json','official_val_prediction_index.json'):
        require(digest(folder/name) == digest(ref/name), 'P2 VAL metadata differs: '+name)
    idx = read(ref/'official_val_prediction_index.json')
    ids = read(HERE/'references/val_images.json')['records']
    records = {x['source_image']:x for x in ids}
    require(len(records) == idx['images_per_method'] == 548, 'VAL image coverage differs')
    report = {'methods':{}, 'prediction_check':'all P2 payloads hashed and every row validated against the frozen per-image count',
              'candidate_counts':{}}
    for method, entries in idx['methods'].items():
        require(len(entries) == 548 and {x['source_image'] for x in entries} == set(records), 'P2 VAL coverage differs')
        for entry in entries:require(safe(folder,entry['path']).is_file(), 'VAL prediction missing: '+entry['path'])
        for number,entry in enumerate(entries,1):
            p = safe(folder,entry['path']);require(digest(p) == entry['sha256'], 'VAL prediction digest differs')
            body = read(p);r = records[entry['source_image']]
            require(body['protocol_sha256']==idx['protocol_sha256'], 'P2 prediction protocol differs')
            validate_p2_rows(body,{'width':r['width'],'height':r['height'],'image_sha256':r['sha256']},method,entry)
            if number in (1,len(entries)) or number%100==0:
                print('VAL source validation '+method+': '+str(number)+'/'+str(len(entries)),flush=True)
        report['methods'][method] = len(entries)
        report['candidate_counts'][method]={'images_validated':len(entries),'max':max(x['detections'] for x in entries),
                                           'images_above_1000':sum(x['detections']>1000 for x in entries)}
    ann_records = [{'annotation':Path(x['source_image']).stem+'.txt','stem':Path(x['source_image']).stem}
                   for x in ids]
    report['annotations'] = annotation_check(project/'data/prepared/visdrone_annotations/val',
                                             ann_records,{Path(x['source_image']).stem:x['annotation_sha256'] for x in ids})
    old = project/'experiments/official_val_a025_v1_seed42'
    frozen = read(HERE/'references/legacy_frozen.json')
    require(digest(old/'protocol.json') == frozen['original_protocol_file_sha256'], 'Original VAL protocol differs')
    legacy_idx = read(old/'prediction_index.json')
    require(legacy_idx['protocol_sha256'] == frozen['original_protocol_content_sha256'] and
            len(legacy_idx['records']) == 548 and
            {x['source_image'] for x in legacy_idx['records']} == set(records), 'Original cascade VAL index differs')
    for item in legacy_idx['records']:
        require(safe(old,item['path']).is_file(), 'Original cascade prediction missing')
        require(item['annotation_sha256'] == records[item['source_image']]['annotation_sha256'],
                'Original cascade/VAL label identity differs')
    for item in frozen['val_parity']:
        require(digest(safe(old,item['original']['path'])) == item['original']['sha256'],
                'Original cascade reference prediction differs')
        body = read(safe(old,item['original']['path']))
        validate_rows({'image_sha256':body['image_sha256'],'detections':body['hybrid_a025']},item['record'])
    report['methods']['hybrid800_legacy'] = 548
    report['original_hycado800_samples_verified'] = len(frozen['val_parity'])
    report['scope'] = 'P2 and original HyCADO800 final predictions; global/grid/relocation VAL adapters remain optional.'
    return report

def run(project, dataset=None, deep=False):
    result = {'id':'hycado_step21_readonly_input_inventory_v1','project':str(project),
              'training':False,'inference':False,'scoring':False,'package_install':False,
              'database_readonly':True,'deep_hash_requested':deep,'sections':{},'issues':[]}
    try:
        manifest = read(HERE/'package_sha256.json')
        for name, expected in manifest.items():
            require(digest(HERE/name) == expected, 'Package file differs: '+name)
        records_doc = read(HERE/'references/stage18/dataset_manifest.json')
        records = [dict(x, image_sha256=x['image_sha256']) for x in records_doc['records']]
        require(len(records) == 1610 and len({x['stem'] for x in records}) == 1610, 'Test-dev records differ')
        sample_info = {}
        for stage in (18,20):
            try:
                folder,p,result_doc,index,sig = checked_reference(stage,project)
                sample_info[stage] = folder
                section = database_check(folder/'predictions.sqlite3',sig,records,index,p['methods'],deep,stage==20)
                if deep:require(section['sha256'] == result_doc['prediction_database_sha256'],'Full DB hash differs')
                result['sections']['stage'+str(stage)] = section
            except Exception as exc:result['issues'].append({'section':'stage'+str(stage),'error':str(exc)})
        try:
            old = project/'experiments'/RUNS[18]
            target = locate_dataset(old, records_doc, dataset)
            expected = {x['stem']:x['sha256'] for x in read(HERE/'references/stage18/annotation_index.json')['records']}
            anns = annotation_check(target/'annotations',records,expected)
            for rec in records:require((target/'images'/(rec['stem']+'.jpg')).is_file(),'Test-dev JPEG missing')
            for rec in samples(records):
                require(digest(target/'images'/(rec['stem']+'.jpg')) == rec['image_sha256'],'JPEG sample SHA differs')
            result['sections']['testdev_dataset'] = {'path':str(target),'annotations':anns,'images_present':1610,'jpeg_hash_samples':3}
        except Exception as exc:result['issues'].append({'section':'testdev_dataset','error':str(exc)})
        try:result['sections']['val'] = val_check(project)
        except Exception as exc:result['issues'].append({'section':'val','error':str(exc)})
        for label,section in result['sections'].items():
            anns = section.get('annotations',{})
            counts = anns.get('counts',{})
            if counts.get('class_specific_ignored',0) or counts.get('nonpositive_geometry',0):
                result['issues'].append({'section':label,'error':'Annotation policy review required; no silent COCO conversion',
                                         'class_specific_ignored':counts.get('class_specific_ignored',0),
                                         'nonpositive_geometry':counts.get('nonpositive_geometry',0)})
    except Exception as exc:result['issues'].append({'section':'package_or_setup','error':str(exc)})
    result['status'] = 'PREFLIGHT_READY' if not result['issues'] else 'NEEDS_REVIEW'
    result['scope'] = 'All P2 VAL payloads/rows and annotation hashes checked; test-dev payload checks and legacy VAL payload checks are sampled here, with complete test-dev database SHA checked separately by the COCO runner.'
    return result

def save_report(result):
    folder = HERE/'reports'
    folder.mkdir(exist_ok=True)
    data = json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n'
    (folder/'21_onkontrol.json').write_text(data,encoding='utf-8')
    path = HERE/'21_onkontrol_rapor.zip'
    temp = path.with_suffix('.partial')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('21_onkontrol.json',data)
        z.write(HERE/'protocol.json','protocol.json')
    os.replace(temp,path)
    return path

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,default=Path(r'D:\HyCADO_Temiz_Baslangic'))
    ap.add_argument('--dataset')
    ap.add_argument('--deep',action='store_true',help='Optional full database SHA256; reads all database bytes and may take time.')
    args = ap.parse_args()
    require(sys.version_info >= (3,11),'Python 3.11 or newer required')
    result = run(args.project.resolve(),args.dataset,args.deep)
    path = save_report(result)
    print('SONUC:',result['status'])
    for issue in result['issues']:print(issue['section']+': '+issue['error'])
    print('GONDERILECEK RAPOR:',path)
    return 0 if result['status'] == 'PREFLIGHT_READY' else 1

if __name__ == '__main__':
    try:sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
