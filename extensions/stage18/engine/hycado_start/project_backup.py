"""Verified project snapshot to a new sibling folder; never deletes source files."""
from __future__ import annotations
import contextlib
from datetime import datetime
import hashlib
import os
from pathlib import Path
import shutil
import stat
import sys
import traceback
import uuid

from .common import ROOT, sha256, write_json, utc_now
from .stage1 import training_lock
from .crop_loader import read_json

EXCLUDED = {'.venv', '__pycache__', '.git', '.pytest_cache'}
LOCKS = [('stage1_v1_seed42', '.training.lock'), ('stage1_predictions_v1_seed42', '.prediction.lock'),
         ('crop_merge_v1_seed42', '.merge.lock'), ('crop_dataset_v1_seed42', '.crop_data.lock'),
         ('rtdetr_crop_v1_seed42', '.training.lock'), ('development_eval_v1_seed42', '.evaluation.lock'),
         ('development_diagnostics_v1_seed42', '.diagnostic.lock'), ('raw_crop_capture_v1_seed42', '.capture.lock'),
         ('score_ablation_v1_seed42', '.analysis.lock'), ('official_val_a025_v1_seed42', '.evaluation.lock')]


def inventory(root):
    root = Path(root).resolve(); result = []
    for folder, directories, files in os.walk(root, followlinks=False):
        directories[:] = sorted(d for d in directories if d not in EXCLUDED)
        for d in directories:
            p = Path(folder) / d
            if p.is_symlink() or (getattr(p.stat(follow_symlinks=False), 'st_file_attributes', 0) & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)):
                raise ValueError('Baglanti klasoru otomatik takip edilmedi: ' + str(p))
        for name in sorted(files):
            p = Path(folder) / name
            if name.endswith('.lock') or p == root / 'reports/12_yedek_sonuc.json': continue
            if p.is_symlink(): raise ValueError('Sembolik dosya baglantisi takip edilmedi: ' + str(p))
            st = p.stat()
            result.append((p.relative_to(root).as_posix(), st.st_size, st.st_mtime_ns))
    return sorted(result)


def copy_checked(source, destination):
    source, destination = Path(source), Path(destination)
    before = source.stat(); digest = hashlib.sha256(); destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as out:
        while True:
            block = src.read(1024 * 1024)
            if not block: break
            out.write(block); digest.update(block)
        out.flush(); os.fsync(out.fileno())
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Yedeklenirken kaynak degisti: ' + str(source))
    expected = digest.hexdigest()
    if sha256(destination) != expected: raise ValueError('Yedek kopyanin SHA256 degeri farkli')
    shutil.copystat(source, destination)
    return expected


def snapshot(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    if parent == root or parent.is_relative_to(root) or root.is_relative_to(parent):
        raise ValueError('Yedek hedefi kaynak projenin icinde/ustunde olamaz')
    parent.mkdir(parents=True, exist_ok=True)
    files = inventory(root); total = sum(size for _, size, _ in files)
    if not files: raise ValueError('Yedeklenecek dosya yok')
    if shutil.disk_usage(parent).free < total + 1024**3: raise OSError('Yedek icin yeterli bos alan yok')
    identity = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    building = parent / (identity + '_TAMAMLANMADI'); final = parent / identity
    building.mkdir(); records = []; done = 0
    # A failed or interrupted snapshot remains explicitly named TAMAMLANMADI.
    for i, (relative, size, mtime) in enumerate(files, 1):
        source = root / relative
        if (source.stat().st_size, source.stat().st_mtime_ns) != (size, mtime):
            raise ValueError('Kaynak envanterden sonra degisti: ' + relative)
        digest = copy_checked(source, building / root.name / relative)
        records.append({'path': relative, 'bytes': size, 'sha256': digest}); done += size
        if i == 1 or i % 500 == 0 or i == len(files):
            print(f'YEDEK: {i}/{len(files)} dosya; {done/1024**3:.2f}/{total/1024**3:.2f} GiB dogrulandi', flush=True)
    if inventory(root) != files: raise ValueError('Yedek sirasinda kaynak agac degisti; tamamlandi sayilmadi')
    manifest = {'status': 'COMPLETED', 'created_utc': utc_now(), 'source_root': str(root),
                'project_folder': root.name, 'excluded_directories': sorted(EXCLUDED),
                'file_count': len(files), 'total_bytes': total, 'records': records}
    write_json(building / 'backup_manifest.json', manifest); building.rename(final)
    return {'status': 'COMPLETED', 'created_utc': utc_now(), 'source_root': str(root),
            'backup_directory': str(final), 'manifest_sha256': sha256(final / 'backup_manifest.json'),
            'file_count': len(files), 'total_bytes': total, 'all_copied_files_sha256_verified': True}


def verify_receipt(root, required):
    """Verify selected critical live files AND their independent backup copies."""
    root = Path(root).resolve(); receipt = read_json(root / 'reports/12_yedek_sonuc.json')
    if (receipt.get('status') != 'COMPLETED' or receipt.get('source_root') != str(root) or
            receipt.get('all_copied_files_sha256_verified') is not True):
        raise ValueError('Once 12A_YEDEK_AL.cmd basariyla tamamlanmali')
    backup = Path(receipt['backup_directory']); path = backup / 'backup_manifest.json'
    if sha256(path) != receipt['manifest_sha256']: raise ValueError('Yedek manifesti degismis')
    manifest = read_json(path); mapping = {r['path']: r for r in manifest['records']}
    if manifest['source_root'] != str(root) or manifest['status'] != 'COMPLETED': raise ValueError('Yedek kaynak kimligi farkli')
    for relative in sorted(set(required)):
        item = mapping.get(relative)
        if item is None: raise ValueError('Dosya yedekte yok: ' + relative)
        if sha256(root / relative) != item['sha256'] or sha256(backup / manifest['project_folder'] / relative) != item['sha256']:
            raise ValueError('Kaynak/yedek degismis: ' + relative + '; once 12A ile yeni yedek alin')
    return receipt


def main():
    receipt_path = ROOT / 'reports/12_yedek_sonuc.json'
    try:
        print('12A: PROJE YEDEGI. Diger HyCADO komutlari kapali olmali.', flush=True)
        with contextlib.ExitStack() as stack:
            stack.enter_context(training_lock(ROOT / '.backup.lock'))
            for folder, name in LOCKS:
                path = ROOT / 'experiments' / folder
                if path.exists(): stack.enter_context(training_lock(path / name))
            receipt = snapshot(ROOT, ROOT.parent / 'HyCADO_Yedekleri')
            write_json(receipt_path, receipt)
        print('YEDEK TAMAMLANDI:', receipt['backup_directory'], flush=True)
        print('Kopyalanan tum dosyalarin SHA256 degeri dogrulandi.', flush=True)
        print('Sonraki komut: 12B_RESMI_VAL_DEGERLENDIR.cmd', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        traceback.print_exc()
        write_json(receipt_path, {'status': 'ERROR', 'error': str(exc), 'created_utc': utc_now()})
        print('YEDEK TAMAMLANMADI. 12B baslatilmadi.', flush=True)
        return 1


if __name__ == '__main__': raise SystemExit(main())
