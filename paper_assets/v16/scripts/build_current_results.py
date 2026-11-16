"""Verify retained stage22--25 evidence, regenerate publication tables/figures.
No inference, training, annotation changes or metric recomputation. Bootstrap
intervals are checked against their retained paired draws (NumPy required).
"""
import argparse,csv,hashlib,io,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; E=ROOT/'evidence/current'
LABELS={'yolo800':'YOLO-800','hybrid_a025':'HyCADO-800','yolo1280':'YOLO-1280','hybrid1280':'HyCADO-1280','yolo1280_rois800':'HyCADO, regions 800 / global 1280','sahi640':'SAHI-640 + YOLO','sahi800':'SAHI-800 + YOLO','yolo1536':'YOLO-1536','yolo800_repeat':'YOLO-800','hybrid800_legacy':'HyCADO-800','relocated_s0_a025':'Relocation seed 0','relocated_s1_a025':'Relocation seed 1','relocated_s2_a025':'Relocation seed 2'}
CLASSES=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
METRICS=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR']
def get(s,n):return json.loads((E/f'stage{s}'/n).read_bytes())
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def near(x,y):np.testing.assert_allclose(x,y,rtol=0,atol=1e-10)
def csvrows(s,n):return list(csv.DictReader(io.StringIO((E/f'stage{s}'/n).read_text(encoding='utf-8-sig'))))
def table(headers,rows,align=None):
 return '\\begin{tabular}{'+(align or 'l'+'r'*(len(headers)-1))+'}\n\\toprule\n'+' & '.join(headers)+r'\\'+'\n\\midrule\n'+'\n'.join(' & '.join(r)+r'\\' for r in rows)+'\n\\bottomrule\n\\end{tabular}\n'
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');parser.add_argument('--figures',action='store_true');a=parser.parse_args()
 r22=get(22,'result.json');r23=get(23,'result.json');r24=get(24,'result.json');r25=get(25,'result.json');p23=get(23,'point.json');c23=get(23,'intervals.json');c24=get(24,'intervals.json');m24=get(24,'metrics.json')
 assert r22['status']=='COCO_POINT_ESTIMATES_COMPLETE' and r22['tests_passed']==13
 assert fp(r22['config'])==r22['signature']
 assert r23['status']=='COCO_UNCERTAINTY_RECALL_COMPLETE' and r23['source_files_unchanged'] and r23['tests_passed']==14
 assert fp(r23['source_bindings'])==r23['signature'] and fp(r22)==r23['source_bindings']['source22_result_fingerprint']
 assert hashlib.sha256((E/'stage23/protocol.json').read_bytes()).hexdigest()==r23['source_bindings']['protocol_sha256']
 assert r24['status']=='SAHI640_TESTDEV_COMPLETE' and r24['original_sources_unchanged'] and r24['tests_passed']==13
 assert fp(r24['runtime_bindings'])==r24['signature']
 assert hashlib.sha256((E/'stage24/protocol.json').read_bytes()).hexdigest()==r24['runtime_bindings']['scientific_protocol_sha256']
 assert fp(r22)==r24['runtime_bindings']['reference22_fingerprint'] and r23['signature']==r24['runtime_bindings']['reference23_signature']
 assert r25['status']=='QUALITATIVE_EXPORT_COMPLETE' and r25['images']==12 and r25['tests_passed']==13
 assert r25['source_files_opened_readonly'] and r25['selected_prediction_payload_hashes_verified'] and not r25['image_selection_uses_predictions']
 metrics=r22['metrics'];td=dict(metrics['testdev']);td['sahi640']=m24['sahi640_coco'];val=metrics['val']
 for m in ['yolo1280','hybrid1280']:assert td[m]==m24['comparators_coco'][m]
 for split,values in [('val',val),('testdev',td)]:
  for m,caps in values.items():
   for cap,v in caps.items():
    assert [x['category_id'] for x in v['per_class']]==list(range(1,11))
    near(v['AP'],np.mean([x['AP'] for x in v['per_class']]))
    assert all(0<=v[k]<=100 for k in METRICS)
 def draws(s):
  ans={}
  for r in csvrows(s,'bootstrap_draws.csv'):
   m=r['method'];i=int(r['draw']);assert i not in ans.setdefault(m,{})
   ans[m][i]={k:float(v) for k,v in r.items() if k not in ['draw','method']}
  assert all(set(v)==set(range(1000)) for v in ans.values());return ans
 d23=draws(23);d24=draws(24)
 for m in ['yolo1280','hybrid1280']:
  for i in range(1000):
   for k in ['AP','AP50','AP75','AR100']:near(d23[m][i][k],d24[m][i][k])
 checks=0
 for ci,dr in [(c23,d23),(c24,d24)]:
  for contrast,vals in ci['differences_pp'].items():
   m,n=contrast.split('_minus_')
   for k,entry in vals.items():
    key='AR' if k=='AR100' else k;near(entry['point'],td[m]['100'][key]-td[n]['100'][key])
    samples=np.array([dr[m][i][k]-dr[n][i][k] for i in range(1000)])
    near([entry['lower95'],entry['upper95']],np.quantile(samples,[.025,.975],method='linear'));near(entry['bootstrap_SE'],samples.std(ddof=1));checks+=1
 for ti,t in enumerate([.5,.75]):
  for bi,b in enumerate([.5,1.,2.,5.,10.]):
   key=f'recall_iou{t}_fp{b}'
   for mi,m in enumerate(p23['methods']):
    v=c23['recall_at_FP'][str(t)][str(b)]['methods'][m]
    near(v['point'],p23['recall_at_FP'][mi][ti][bi]);near([v['lower95'],v['upper95']],np.quantile([d23[m][i][key] for i in range(1000)],[.025,.975]))
   for name,v in c23['recall_at_FP'][str(t)][str(b)]['differences_pp'].items():
    m,n=name.split('_minus_');samples=np.array([d23[m][i][key]-d23[n][i][key] for i in range(1000)])
    near([v['lower95'],v['upper95']],np.quantile(samples,[.025,.975]));near(v['bootstrap_SE'],samples.std(ddof=1));checks+=1
 headers=['Method','AP',r'AP$_{50}$',r'AP$_{75}$',r'AP$_S$',r'AP$_M$',r'AP$_L$',r'AR$_{100}$']
 outputs={}
 order=['yolo800','hybrid_a025','yolo1280','hybrid1280','yolo1280_rois800','sahi640']
 outputs['coco_testdev_main']=table(headers,[[LABELS[m]]+[f'{td[m]["100"][k]:.3f}' for k in METRICS] for m in order])
 for split,values in [('val',val),('testdev',td)]:
  for cap in ['100','500']:
   hs=headers.copy();hs[-1]=r'AR$_{'+cap+'}$'
   outputs[f'coco_{split}_{cap}']=table(hs,[[LABELS[m]]+[f'{v[cap][k]:.3f}' for k in METRICS] for m,v in values.items()])
 contrasts=[('HyCADO-1280 $-$ YOLO-1280',c23,'hybrid1280_minus_yolo1280'),('HyCADO-1280 $-$ SAHI-640',c24,'hybrid1280_minus_sahi640'),('SAHI-640 $-$ YOLO-1280',c24,'sahi640_minus_yolo1280'),('Regions 800 / global 1280 $-$ YOLO-1280',c23,'yolo1280_rois800_minus_yolo1280'),('HyCADO-1280 $-$ regions 800 / global 1280',c23,'hybrid1280_minus_yolo1280_rois800')]
 cirows=[]
 for label,ci,key in contrasts:
  for k in ['AP','AP50','AP75','AR100']:
   v=ci['differences_pp'][key][k];cirows.append([label,k,f'{v["point"]:+.3f}',f'[{v["lower95"]:+.3f}, {v["upper95"]:+.3f}]'])
 outputs['coco_all_intervals']=table(['Contrast','Metric',r'$\Delta$',r'95\% interval'],cirows,'llrr')
 outputs['coco_class_testdev']=table(['Class','YOLO-1280','HyCADO-1280','SAHI-640',r'$\Delta$H--Y',r'$\Delta$H--S'],[[c]+[f'{td[m]["100"]["per_class"][i]["AP"]:.3f}' for m in ['yolo1280','hybrid1280','sahi640']]+[f'{td["hybrid1280"]["100"]["per_class"][i]["AP"]-td[m]["100"]["per_class"][i]["AP"]:+.3f}' for m in ['yolo1280','sahi640']] for i,c in enumerate(CLASSES)])
 br=[]
 for t in [.5,.75]:
  for b in [.5,1.,2.,5.,10.]:
   v=c23['recall_at_FP'][str(t)][str(b)];r=v['methods']['yolo1280'];d=v['differences_pp']['hybrid1280_minus_yolo1280']
   br.append([f'{t:.2f}',f'{b:g}',f'{r["point"]:.3f}',f'[{r["lower95"]:.3f}, {r["upper95"]:.3f}]',f'{d["point"]:+.3f}',f'[{d["lower95"]:+.3f}, {d["upper95"]:+.3f}]'])
 outputs['testdev_recall_budgets']=table(['IoU','FP/image','Recall', r'95\% interval',r'$\Delta$H--Y',r'95\% interval for $\Delta$'],br,'rrrrrr')
 st=m24['sahi640_toolkit'];outputs['sahi_toolkit_testdev']=table(['Method',r'$\APlit$','AP50','AP75','AR500'],[[LABELS[m]]+[f'{v[k]:.3f}' for k in ['AP','AP50','AP75','AR500']] for m,v in list(m24['comparators_toolkit'].items())+[('sahi640',st)]])
 for name,content in outputs.items():
  path=ROOT/'tables'/f'{name}.tex'
  if a.write:path.write_text(content,encoding='utf-8')
  else:assert path.read_text(encoding='utf-8')==content,name
 # Primary prose numbers are macros generated from exactly the same records.
 scalar_values={'CocoHY':c23['differences_pp']['hybrid1280_minus_yolo1280']['AP']['point'],'CocoHYLow':c23['differences_pp']['hybrid1280_minus_yolo1280']['AP']['lower95'],'CocoHYHigh':c23['differences_pp']['hybrid1280_minus_yolo1280']['AP']['upper95'],'CocoHS':c24['differences_pp']['hybrid1280_minus_sahi640']['AP']['point'],'CocoHSLow':c24['differences_pp']['hybrid1280_minus_sahi640']['AP']['lower95'],'CocoHSHigh':c24['differences_pp']['hybrid1280_minus_sahi640']['AP']['upper95'],'CocoY':td['yolo1280']['100']['AP'],'CocoH':td['hybrid1280']['100']['AP'],'CocoS':td['sahi640']['100']['AP']}
 content=''.join('\\newcommand{\\'+k+'}{'+f'{v:.3f}'+'}\n' for k,v in scalar_values.items());path=ROOT/'tables/coco_scalars.tex'
 if a.write:path.write_text(content)
 else:assert path.read_text()==content
 if a.figures:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
  fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.0),gridspec_kw={'width_ratios':[1,1.16]})
  fig.subplots_adjust(left=.22,right=.985,bottom=.18,top=.77,wspace=.91)
  labs=['HyCADO-1280\n− YOLO-1280','HyCADO-1280\n− SAHI-640','SAHI-640\n− YOLO-1280']
  for i,(_,ci,key) in enumerate(contrasts[:3]):
   v=ci['differences_pp'][key]['AP'];y=2-i
   ax.errorbar(v['point'],y,xerr=[[v['point']-v['lower95']],[v['upper95']-v['point']]],fmt='o',markersize=4,capsize=3,color=['#0072B2','#D55E00','#666666'][i],lw=1.3)
  ax.set_yticks([2,1,0],labs,fontsize=7.3);ax.tick_params(axis='y',length=0)
  ax.set_ylim(-.55,2.55);ax.set_xlim(-.9,1.3);ax.set_xticks([-.5,0,.5,1]);ax.axvline(0,c='#555555',ls=':',lw=.8);ax.grid(axis='x',alpha=.17)
  ax.set_xlabel('COCO-style AP difference (points)',fontsize=7.1);ax.set_title('(a) Paired 95% intervals',fontsize=8.5,pad=28)
  y=np.arange(10);dh=np.array([x['AP'] for x in td['hybrid1280']['100']['per_class']])
  for offset,m,label,col in [(-.18,'yolo1280','H − YOLO-1280','#0072B2'),(.18,'sahi640','H − SAHI-640','#D55E00')]:
   bx.barh(y+offset,dh-np.array([x['AP'] for x in td[m]['100']['per_class']]),height=.32,color=col,label=label)
  bx.set_yticks(y,CLASSES,fontsize=7.2);bx.invert_yaxis();bx.axvline(0,color='#555555',lw=.8);bx.set_xlim(-3.2,2.2);bx.set_xticks([-3,-2,-1,0,1,2]);bx.grid(axis='x',alpha=.17)
  bx.set_xlabel('Class AP difference (points)',fontsize=7.1);bx.set_title('(b) Class-wise point differences',fontsize=8.5,pad=28)
  bx.legend(loc='lower center',bbox_to_anchor=(.5,1.005),ncol=2,fontsize=6.5,frameon=False,handlelength=1,columnspacing=.9)
  for ext in ['pdf','png','svg']:fig.savefig(ROOT/'figures'/f'coco_comparison.{ext}',dpi=240)
  plt.close(fig)
  curves=csvrows(23,'recall_curves.csv');fig,axs=plt.subplots(1,2,figsize=(7.15,2.65),sharex=True,sharey=True)
  colors=['#727272','#0072B2','#D55E00'];styles=['solid',(0,(6,3)),(0,(1,2))]
  for ax,t in zip(axs,[.5,.75]):
   for i,m in enumerate(p23['methods']):
    rs=[r for r in curves if r['method']==m and float(r['IoU'])==t];x=np.array([float(r['FP_per_image']) for r in rs]);y=np.array([float(r['recall_percent']) for r in rs])
    near(x,np.array([int(r['FP']) for r in rs])/1610);near(y,100*np.array([int(r['TP']) for r in rs])/74707)
    for bi,b in enumerate([.5,1.,2.,5.,10.]):near(y[np.searchsorted(x,b,side='right')-1],p23['recall_at_FP'][i][[.5,.75].index(t)][bi])
    ax.step(x,y,where='post',color=colors[i],lw=[3.2,2,1.2][i],ls=styles[i],label=['YOLO-1280','HyCADO-1280','Regions 800 / global 1280'][i])
   ax.set_xlim(0,10);ax.set_ylim(0,60);ax.set_xticks([0,2,4,6,8,10]);ax.grid(alpha=.17);ax.set_title(f'IoU = {t:.2f}',fontsize=9);ax.set_xlabel('False positives per image')
  axs[0].set_ylabel('Pooled recall (%)');h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=7.5)
  fig.tight_layout(rect=(0,.1,1,1))
  for ext in ['pdf','png','svg']:fig.savefig(ROOT/'figures'/f'recall_testdev.{ext}',dpi=240)
  plt.close(fig)
 verify={'status':'PASS','source22_23_24_signature_bindings_verified':True,'all_exported_paired_COCO_and_recall_intervals_recomputed_from_retained_draws':True,'contrast_metric_checks':checks,'stage24_comparator_draws_equal_stage23':True,'generated_tables':len(outputs)+1,'new_training':False,'new_inference':False,'new_detection_metric_computation':False,'qualitative_source_export_tests':r25['tests_passed']}
 
 if a.write or a.figures:(E/'v14_numeric_validation.json').write_text(json.dumps(verify,indent=2)+'\n')
 print(json.dumps(verify))
if __name__=='__main__':main()
