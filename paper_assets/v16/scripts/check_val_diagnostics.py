"""Verify frozen VAL reports and regenerate v11 tables without rerunning inference."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'evidence'
A=E/'val_diagnostics'
HASHES={
    '16_val_boyut_yogunluk_rapor.zip':'cf6a5ab170360e25be0d169be08755da441c274b2e30b9e8bd9a8656cf3bc989',
    '17_val_esit_yanlis_alarm_rapor.zip':'8859646f4a1b0bb34e7527ce3e03a1b4bd776ab40eaffd849691b5d179e3565c',
    'HyCADO_16_VAL_Boyut_Yogunluk_CPU.zip':'57de402eca198d7fb2e10e2a8a010e4113e65b075e87aa36610be166e0ff9aad',
    'HyCADO_17_Esit_Yanlis_Alarm_CPU.zip':'ad3af76cbe7b1c01f73f5d6de628bca4797556223cd0fd511949469dfd1c4532',
}
M=('yolo1280','hycado_v1')
ROW=r'\\'+'\n'


def need(ok,label):
    if not ok: raise ValueError(label)


def close(x,y):
    return math.isclose(x,y,rel_tol=0,abs_tol=1e-9)


def zipped_report(name,csvs):
    with zipfile.ZipFile(A/name) as z:
        need(z.testzip() is None,name)
        d=json.loads(z.read('result.json'))
        need(d['status']=='COMPLETE' and d['images']==548 and
             d['inference'] is False and d['bootstrap_draws']==2000,name)
        for key,member in csvs:
            rows=list(csv.DictReader(io.StringIO(z.read(member).decode('utf-8-sig'))))
            need(len(rows)==len(d[key]),f'{name} {member} length')
            for csv_row,js_row in zip(rows,d[key]):
                need(csv_row=={k:'' if v is None else str(v) for k,v in js_row.items()},f'{name} {member} values')
        return d


def pp(x):
    if x is None: return '--'
    rounded=f'{x:+.2f}'
    return '$'+('0.00' if rounded in ('+0.00','-0.00') else rounded)+'$'


def interval(ci):
    return r'$['+f'{ci[0]:.2f}, {ci[1]:.2f}'+r']$' if ci else '--'


def pct(x):
    return f'{x:.2f}' if x is not None else '--'


def table(cols,head,rows):
    return (r'\begin{tabular}{'+cols+'}\n'+r'\toprule'+'\n'+head+ROW+r'\midrule'+'\n'+
            ''.join(' & '.join(map(str,r))+ROW for r in rows)+r'\bottomrule'+'\n'+r'\end{tabular}'+'\n')


def build():
    for name,expected in HASHES.items():
        need(hashlib.sha256((A/name).read_bytes()).hexdigest()==expected,name)
    sizes=zipped_report('16_val_boyut_yogunluk_rapor.zip',
                        [('size_rows','size_rows.csv'),('density_rows','density_rows.csv')])
    fp=zipped_report('17_val_esit_yanlis_alarm_rapor.zip',[('rows','matched_fp_operating_points.csv')])
    need(sizes['cpu_only'] and not sizes['trained'] and not fp['gpu'] and not fp['training'],'CPU-only diagnostics')
    need(sizes['verified_detection_totals']==fp['verified_detection_totals'],'Archived count parity')
    old=json.loads((E/'results.json').read_text())['metrics']['hybrid_a025']
    y=json.loads((E/'p2_val/official_val_metrics.json').read_text())['methods']['yolo1280']
    sd={(r['iou'],r['area_px2']):r for r in sizes['size_rows']}
    dn={(r['iou'],r['normal_gt_per_image']):r for r in sizes['density_rows']}
    ft={(r['stratum'],r['iou'],r['fp_per_image_budget']):r for r in fp['rows']}
    area_labels=[('<256',r'$<256$'),('[256,1024)',r'$[256,1024)$'),
                 ('[1024,9216)',r'$[1024,9216)$'),('>=9216',r'$\geq 9216$')]
    density_labels=[('1-30','1--30'),('31-100','31--100'),('101+','101 or more')]
    strata=[('all','All VAL'),('1-30_GT','1--30 GT objects')]
    budgets=(.5,1.,2.,5.,10.)
    need(len(sd)==len(sizes['size_rows'])==10 and len(dn)==len(sizes['density_rows'])==8 and
         len(ft)==len(fp['rows'])==20,'Diagnostic table coverage')
    for t in ('0.5','0.75'):
        a=[sd[t,k] for k,_ in area_labels]; d=[dn[t,k] for k,_ in density_labels]
        need(sd[t,'nonpositive']['normal_gt']==0 and dn[t,'0']['images']==0 and
             sum(r['images'] for r in d)==548 and
             sum(r['normal_gt'] for r in a)==sum(r['normal_gt'] for r in d)==38599,'Full VAL accounting')
        for m,reference in zip(M,(y,old)):
            ref=reference['counts_at_fixed_max500'][t]
            need(ref==fp['verified_detection_totals'][t][m] and
                 sum(r[m+'_matched'] for r in a)==sum(r[m+'_matched'] for r in d)==ref['TP'] and
                 sum(r[m+'_fp'] for r in d)==ref['FP'],f'{t} {m} totals')
        for r in a+d:
            n=r['normal_gt']; yr=r['yolo1280_matched']; hr=r['hycado_v1_matched']
            need(close(r['recall_delta_pp'],100*(hr-yr)/n) and
                 close(r['yolo1280_recall_pct'],100*yr/n) and close(r['hycado_v1_recall_pct'],100*hr/n) and
                 len(r['recall_delta_ci95_pp'])==2 and r['bootstrap_valid_draws']==2000,'Stratified recalls')
    for (s,t,b),r in ft.items():
        need(s in dict(strata) and t in (.5,.75) and b in budgets and
             r['primary']==(t==.5 and b==2.) and r['joint_budget_support'] and
             r['bootstrap_valid_draws']==2000 and len(r['recall_delta_ci95_pp'])==2,'Matched budget coverage')
        nr,ng=(548,38599) if s=='all' else (dn['0.5','1-30']['images'],dn['0.5','1-30']['normal_gt'])
        need((r['images'],r['normal_gt'])==(nr,ng),'Stratum identity')
        for m in M:
            need(close(r[m+'_recall_pct'],100*r[m+'_tp']/ng) and
                 close(r[m+'_fp_per_image'],r[m+'_fp']/nr) and r[m+'_fp_per_image']<=b,
                 'Matched budget recall and FP cap')
        need(close(r['recall_delta_pp'],r['hycado_v1_recall_pct']-r['yolo1280_recall_pct']),'Paired delta')
    primary=[ft[s,.5,2.] for s,_ in strata]
    main_rows=[[label,pct(r[M[0]+'_recall_pct']),pct(r[M[1]+'_recall_pct']),
                pp(r['recall_delta_pp']),interval(r['recall_delta_ci95_pp'])]
               for (_,label),r in zip(strata,primary)]
    budget_rows=[]
    for s,label in strata:
        for t in (.5,.75):
            for b in budgets:
                r=ft[s,t,b]
                budget_rows.append([label if (t,b)==(.5,.5) else '',f'{t:.2f}' if b==.5 else '',f'{b:g}',
                    *(f"{r[m+'_fp_per_image']:.3f}" for m in M),*(pct(r[m+'_recall_pct']) for m in M),
                    pp(r['recall_delta_pp']),interval(r['recall_delta_ci95_pp'])])
    size_rows=[]
    density_rows=[]
    for t in ('0.5','0.75'):
        for i,(key,label) in enumerate(area_labels):
            r=sd[t,key]
            size_rows.append([f'{float(t):.2f}' if i==0 else '',label,r['normal_gt'],
                              *(pct(r[m+'_recall_pct']) for m in M),
                              pp(r['recall_delta_pp']),interval(r['recall_delta_ci95_pp'])])
        for i,(key,label) in enumerate(density_labels):
            r=dn[t,key]
            density_rows.append([f'{float(t):.2f}' if i==0 else '',label,r['images'],r['normal_gt'],
                                 pp(r['recall_delta_pp']),interval(r['recall_delta_ci95_pp']),
                                 *(pct(r[m+'_precision_pct']) for m in M),*(r[m+'_fp'] for m in M)])
    return {
        'matched_fp_main.tex':table('lrrrr',
            r'Group & YOLO & HyCADO & $\Delta$ & 95\% interval',main_rows),
        'matched_fp_full.tex':table('lrrrrrrrr',
            r'Group & IoU & Cap & YOLO FP/I & HyCADO FP/I & YOLO R & HyCADO R & $\Delta$ & 95\% interval',budget_rows),
        'val_size_recall.tex':table('llrrrrr',
            r'IoU & Area (px$^2$) & GT & YOLO R & HyCADO R & $\Delta$ & 95\% interval',size_rows),
        'val_density_recall.tex':table('llrrrrrrrr',
            r'IoU & GT/image & Images & GT & $\Delta$R & 95\% interval & YOLO P & HyCADO P & YOLO FP & HyCADO FP',density_rows),
    }


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--write-tables',action='store_true')
    args=ap.parse_args()
    for file,data in build().items():
        p=ROOT/'tables'/file
        if args.write_tables: p.write_text(data,encoding='utf-8')
        else: need(p.read_text(encoding='utf-8')==data,file)
    print('Verified both frozen diagnostics, VAL counts, all 20 operating points, and 4 typeset tables.')
