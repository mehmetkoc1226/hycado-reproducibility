"""Verify completed28--30 reports and reproduce v16 controls tables/figures.

No training, inference, matching or new AP calculation. Uses retained point
metrics and paired draws. --write updates derived tables; --figures needs matplotlib.
"""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent

def readz(z,name):return json.loads(z.read(name))
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def need(v,message):
    if not v:raise ValueError(message)

def load():
    reports={}
    for stage,name,status in [(28,'28_cift_cozunurluk_fuzyon_rapor.zip','FUSION28_COMPLETE'),
                              (29,'29_ham_crop_denetim_rapor.zip','RAW29_AUDIT_COMPLETE'),
                              (30,'30_alpha_duyarlilik_rapor.zip','ALPHA30_COMPLETE')]:
        with zipfile.ZipFile(ROOT/'evidence/current'/name) as z:
            r=readz(z,'result.json');b=readz(z,'source_bindings.json')
            need(r['status']==status and r['images']==1610,'Completion/coverage differs')
            need(fp(b['config'])==b['signature']==r['signature'],'Source binding signature differs')
            m=readz(z,'code/package_sha256.json')
            need(hashlib.sha256(z.read('code/package_sha256.json')).hexdigest()==b['config']['package_manifest_sha256'],'Package manifest differs')
            for n in z.namelist():
                if n.startswith('code/') and n not in ['code/package_sha256.json']:
                    need(hashlib.sha256(z.read(n)).hexdigest()==m[n[5:]],'Retained code hash differs')
            reports[stage]={'result':r}
            if stage==29:
                idx=readz(z,'verified_raw_index.json');need(fp(idx)==r['verified_raw_index_sha256'],'Raw index signature differs')
                reports[stage]['geometry']=readz(z,'geometry_summary.json')
                need(all(x['exact_HyCADO800_replay'] and x['exact_HyCADO1280_replay'] and x['exact_mixed_resolution_replay'] for x in idx['records']),'Incomplete exact replay')
            else:
                idx=readz(z,'prediction_index.json');need(fp(idx)==r['prediction_freeze_sha256'],'Prediction freeze signature differs')
                reports[stage]['point']=readz(z,'point_results.json')['metrics']
            need(len(idx['records'])==len({x['image'] for x in idx['records']})==1610,'Index image coverage differs')
            if stage==28:
                reports[stage]['intervals']=readz(z,'intervals.json')
                rows=list(csv.DictReader(io.StringIO(z.read('bootstrap_draws.csv').decode('utf-8-sig'))));draws={}
                keys=['AP','AP50','AP75','AR100']
                for method in ['yolo1280','hybrid1280','sahi640','yolo800_1280_fusion']:
                    selected=[x for x in rows if x['method']==method];d={int(x['draw']):x for x in selected}
                    need(len(selected)==len(d)==1000 and set(d)==set(range(1000)),'Draw coverage differs')
                    draws[method]=np.array([[float(d[i][k]) for k in keys] for i in range(1000)])
                for name,values in reports[stage]['intervals']['differences_pp'].items():
                    left,right=name.split('_minus_')
                    for j,key in enumerate(keys):
                        delta=draws[left][:,j]-draws[right][:,j];q=np.quantile(delta,[.025,.975],method='linear');v=values[key]
                        need(np.allclose(q,[v['lower95'],v['upper95']],atol=1e-12,rtol=0),'Paired interval differs')
                        need(abs(delta.std(ddof=1)-v['bootstrap_SE'])<1e-12,'Paired SE differs')
                        pk='AR' if key=='AR100' else key
                        point=reports[stage]['point'][left]['100'][pk]-reports[stage]['point'][right]['100'][pk]
                        need(abs(point-v['point'])<1e-12,'Paired point difference differs')
            if stage==30:
                reports[stage]['geometry']=readz(z,'geometry_matching.json')
                need(r['alpha025_max_abs_difference_pp']==0 and not r['final_alpha_changed'],'Original alpha0.25 changed')
                for groups in reports[stage]['geometry']['counts'].values():
                    for prefix in ['crop','display_crop']:
                        for k in ['TP','FP','ignored']:need(groups[prefix+'_all'][k]==groups[prefix+'_ratio_ge_0_95'][k]+groups[prefix+'_ratio_lt_0_95'][k],'Geometry partition differs')
                rows=list(csv.DictReader(io.StringIO(z.read('alpha_metrics.csv').decode('utf-8-sig'))));need(len(rows)==10,'Alpha metric coverage differs')
                for x in rows:
                    for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']:
                        need(abs(float(x[k])-reports[stage]['point'][x['method']][x['maxDets']][k])<1e-12,'Alpha CSV/point differs')
    for stage in [28,30]:
        for caps in reports[stage]['point'].values():
            for vals in caps.values():need(abs(np.mean([x['AP'] for x in vals['per_class']])-vals['AP'])<1e-10,'Class AP average differs')
    return reports

def table(headers,rows,spec):
    return '\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(headers)+'\\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+'\\\\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n'

def derived(d):
    p=d[28]['point'];fuse=p['yolo800_1280_fusion'];h=p['hybrid1280'];y=p['yolo1280'];s=p['sahi640'];ci=d[28]['intervals']['differences_pp'];new={}
    original=(ROOT/'tables/coco_testdev_main.tex').read_text()
    vals=fuse['100'];row='YOLO-800 + YOLO-1280 fusion & '+' & '.join(f'{vals[k]:.3f}' for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR'])+'\\\\\n'
    new['coco_testdev_main_v16.tex']=original.replace('\\bottomrule',row+'\\bottomrule')
    names=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor'];rows=[]
    for i,name in enumerate(names):
        yy,hh,ss,ff=[m['100']['per_class'][i]['AP'] for m in [y,h,s,fuse]]
        rows.append([name]+[f'{v:.3f}' for v in [yy,hh,ss,ff]]+[f'{v:+.3f}' for v in [hh-yy,hh-ss,hh-ff]])
    new['coco_class_v16.tex']=table(['Class','YOLO-1280','HyCADO-1280','SAHI-640','Fusion','$\\Delta$H--Y','$\\Delta$H--S','$\\Delta$H--F'],rows,'lrrrrrrr')
    rows=[]
    for m,alpha in [('a000',0),('a0125',.125),('a025',.25),('a050',.5),('a100',1)]:
        for cap in ['100','500']:
            v=d[30]['point'][m][cap];rows.append([f'{alpha:g}',cap]+[f'{v[k]:.3f}' for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']])
    new['alpha_points.tex']=table(['$\\alpha$','maxDets','AP','AP$_{50}$','AP$_{75}$','AP$_S$','AP$_M$','AP$_L$','AR'],rows,'rrrrrrrrr')
    rows=[]
    labels={'hybrid1280_minus_yolo800_1280_fusion':'HyCADO-1280 $-$ fusion','yolo800_1280_fusion_minus_yolo1280':'Fusion $-$ YOLO-1280','yolo800_1280_fusion_minus_sahi640':'Fusion $-$ SAHI-640'}
    for name,metrics in ci.items():
        for k,v in metrics.items():rows.append([labels[name],k,f"{v['point']:+.3f}",f"[{v['lower95']:+.3f}, {v['upper95']:+.3f}]"])
    new['fusion_intervals.tex']=table(['Contrast','Metric','$\\Delta$','95\\% interval'],rows,'llrr')
    rows=[]
    for cap in ['100','500']:
        v=fuse[cap];rows.append([cap]+[f'{v[k]:.3f}' for k in ['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']])
    new['fusion_points.tex']=table(['maxDets','AP','AP$_{50}$','AP$_{75}$','AP$_S$','AP$_M$','AP$_L$','AR'],rows,'rrrrrrrr')
    rows=[]
    for iou,groups in d[30]['geometry']['counts'].items():
        for key,label in [('crop_all','All crop'),('crop_ratio_ge_0_95','Area ratio $\\geq0.95$'),('crop_ratio_lt_0_95','Area ratio $<0.95$'),('display_crop_all','Display-score crop'),('display_crop_ratio_ge_0_95','Display, ratio $\\geq0.95$'),('display_crop_ratio_lt_0_95','Display, ratio $<0.95$')]:
            v=groups[key];rows.append([iou,label,str(v['TP']),str(v['FP']),str(v['ignored']),f"{100*v['precision_in_this_diagnostic']:.3f}"])
    new['crop_geometry_matches.tex']=table(['IoU','Crop subset','TP','FP','Ignored','Precision (\\%)'],rows,'llrrrr')
    return new

def figures(d):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained')
    x=np.array([0,.125,.25,.5,1.]);methods=['a000','a0125','a025','a050','a100']
    for ax,key,title in zip(axes,['AP','AP_small'],['(a) Overall COCO AP','(b) Small-object COCO AP']):
        yy=[d[30]['point'][m]['100'][key] for m in methods]
        ax.plot(x,yy,'o-',color='#186a94',lw=1.6,ms=4,label='HyCADO sensitivity')
        ax.plot([.25],[yy[2]],'D',color='#d86b16',ms=5,label='Original alpha = 0.25')
        ax.axhline(d[28]['point']['yolo800_1280_fusion']['100'][key],color='#555555',ls='--',lw=1.1,label='YOLO-800 + YOLO-1280')
        ax.set(xlabel=r'Crop-score multiplier $\alpha$',ylabel='AP (points)',title=title)
        ax.set_xticks([0,.25,.5,1],labels=['0','0.25','0.5','1'])
        ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=7,loc='lower left');fig.savefig(ROOT/'figures/alpha_sensitivity.pdf');fig.savefig(ROOT/'figures/alpha_sensitivity.png',dpi=180);plt.close(fig)
    with zipfile.ZipFile(ROOT/'evidence/current/23_coco_guven_recall_rapor.zip') as z:
        curves=list(csv.DictReader(io.StringIO(z.read('recall_curves.csv').decode('utf-8-sig'))))
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),sharex=True,sharey=True,layout='constrained')
    styles=[('yolo1280','YOLO-1280','#333333','-'),('hybrid1280','HyCADO-1280','#1684aa','--'),('yolo1280_rois800','Regions 800 / global 1280','#df7c27',':')]
    for ax,iou in zip(axes,['0.5','0.75']):
        for m,label,color,ls in styles:
            rr=[x for x in curves if x['method']==m and x['IoU']==iou]
            need(bool(rr),'Missing retained recall curve')
            ax.plot([float(r['FP_per_image']) for r in rr],[float(r['recall_percent']) for r in rr],label=label,color=color,ls=ls,lw=1.4)
        ax.set_xscale('symlog',linthresh=.5);ax.set_xlim(0,500);ax.set_ylim(0,85)
        ax.set_xticks([0,.5,2,10,100,500],labels=['0','0.5','2','10','100','500'])
        ax.set(xlabel='False positives per image',title='IoU = '+iou);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Pooled recall (%)');axes[0].legend(fontsize=7,loc='upper left')
    fig.savefig(ROOT/'figures/recall_testdev_full_symlog.pdf');fig.savefig(ROOT/'figures/recall_testdev_full_symlog.png',dpi=180);plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');ap.add_argument('--figures',action='store_true');args=ap.parse_args()
    d=load();files=derived(d)
    for n,s in files.items():
        p=ROOT/'tables'/n
        if args.write:p.write_text(s)
        else:need(p.read_text()==s,'Generated table differs: '+n)
    if args.figures:figures(d)
    print('PASS: completed28/29/30 bindings,12 paired intervals, alpha metric coverage, class averages, geometry partitions and',len(files),'derived tables. No new inference or metric matching.')

if __name__=='__main__':main()
