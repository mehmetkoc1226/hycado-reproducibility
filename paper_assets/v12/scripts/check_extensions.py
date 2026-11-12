"""Validate completed reports and printed tables, without inference or AP recomputation."""
from pathlib import Path
import json,hashlib,csv,io,math,zipfile,argparse
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'evidence/extensions'
def get(folder,name):return json.loads((E/folder/name).read_bytes())
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def near(a,b):assert abs(a-b)<1e-8,(a,b)
names=['18_testdev_dogrulama_rapor','19_tam1280_gelistirme_rapor','20_testdev1280_rapor']
A,B,C=[get(n,'metrics.json') for n in names]
for folder,n in zip(names,[1610,647,1610]):
 p=get(folder,'protocol.json');r=get(folder,'result.json');m=get(folder,'metrics.json');idx=get(folder,'prediction_index.json');sig=fp(p)
 assert all(x['protocol_sha256']==sig for x in [r,m,idx])
 assert r['status']=='COMPLETED' and r['images']==n
 assert len(idx['records'])==len({x['image'] if 'image' in x else x['source_image'] for x in idx['records']})==n
 assert len(get(folder,'bootstrap_groups.json')['image_to_group'])==n
 for vals in m.get('methods',m.get('variants',{})).values():
  cls=vals['per_class'];den=sum(x['image_presence_count'] for x in cls)
  near(vals['AP'],sum(x['AP']*x['image_presence_count']/den for x in cls))
 for row in csv.DictReader(io.StringIO((E/folder/'metrics.csv').read_text(encoding='utf-8-sig'))):
  if row['method'] not in m['methods']:continue
  for k in ['AP','AP50','AP75','AR1','AR10','AR100','AR500']:near(float(row[k]),m['methods'][row['method']][k])
 assert get(folder,'intervals.json')['valid_replicates']==1000
p=get(names[2],'protocol.json')
for label,folder in [('stage18',names[0]),('stage19',names[1])]:
 for name,digest in p['source_bindings'][label].items():
  if name!='database_sha256':assert hashlib.sha256((E/folder/name).read_bytes()).hexdigest()==digest
assert C['methods']['yolo1280']==A['methods']['yolo1280']
assert C['methods']['hybrid800_reference']==A['methods']['hybrid_a025']
a=A['methods'];c=C['methods'];ci=get(names[2],'intervals.json')['differences_pp']
for contrast,x,y in [('hybrid1280_minus_yolo1280','hybrid1280','yolo1280'),('yolo1280_rois800_minus_yolo1280','yolo1280_rois800','yolo1280'),('hybrid1280_minus_yolo1280_rois800','hybrid1280','yolo1280_rois800')]:
 near(c[x]['AP']-c[y]['AP'],ci[contrast]['AP']['point'])
 for m in ci[contrast].values():assert m['lower95']<=m['point']<=m['upper95']
row=r'\\'+'\n'
table=r'\begin{tabular}{lrrrrr}'+'\n'+r'\toprule'+'\n'+r'Method & $\APlit$ & AP50 & AP75 & AR500 & Macro AP'+row+r'\midrule'+'\n'
for label,key in [('YOLO-1280','yolo1280'),('HyCADO-1280 (1280 regions + 1280 global)','hybrid1280'),('800 regions + 1280 global','yolo1280_rois800'),('HyCADO-800 reference','hybrid800_reference')]:
 table+=' & '.join([label,*[f'{c[key][k]:.3f}' for k in ['AP','AP50','AP75','AR500','class_macro_AP_separate_diagnostic']]])+row
table+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
cit=r'\begin{tabular}{lrr}'+'\n'+r'\toprule'+'\n'+r'Contrast & $\Delta$AP & 95\% interval'+row+r'\midrule'+'\n'
for label,key,sgn in [('1280 regions minus YOLO-1280','hybrid1280_minus_yolo1280',1),('800 regions minus YOLO-1280','yolo1280_rois800_minus_yolo1280',1),('800 minus 1280 regions','hybrid1280_minus_yolo1280_rois800',-1)]:
 d=ci[key]['AP'];point=d['point']*sgn;lo,hi=(d['lower95'],d['upper95']) if sgn==1 else (-d['upper95'],-d['lower95'])
 cit+=f'{label} & {point:+.3f} & [{lo:+.3f}, {hi:+.3f}]'+row
cit+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
cls=r'\begin{tabular}{lrrrrr}'+'\n'+r'\toprule'+'\n'+r'Class & YOLO-1280 & 1280 regions & 800 regions & $\Delta$1280 & $\Delta$800'+row+r'\midrule'+'\n'
class_names=['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor']
for i,name in enumerate(class_names):
 vals=[c[m]['per_class'][i]['AP'] for m in ['yolo1280','hybrid1280','yolo1280_rois800']]
 cls+=' & '.join([name,*[f'{x:.3f}' for x in vals],f'{vals[1]-vals[0]:+.3f}',f'{vals[2]-vals[0]:+.3f}'])+row
cls+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'
parser=argparse.ArgumentParser();parser.add_argument('--write-tables',action='store_true');args=parser.parse_args()
for name,content in [('testdev1280',table),('testdevci',cit),('testdev_per_class',cls)]:
 path=ROOT/'tables'/(name+'.tex')
 if args.write_tables:path.write_text(content)
 else:assert path.read_text()==content,name
# Preserve every original VAL accuracy cell, while excluding timing columns.
orig=json.loads((ROOT/'evidence/results.json').read_bytes())['metrics'];p2=json.loads((ROOT/'evidence/p2_val/official_val_metrics.json').read_bytes())['methods'];merged=orig|p2
keys=['baseline','yolo1280','yolo1536','sahi640','sahi800','hybrid_a025','rtdetr_full960','full_fusion_a100','full_fusion_a025','fixed_grid_a025','relocated_mean']
data=[line for line in (ROOT/'tables/accuracy.tex').read_text().splitlines() if ' & ' in line and not line.startswith('Method')]
assert len(data)==len(keys)
for line,key in zip(data,keys):
 vals=line.removesuffix(r'\\').split(' & ')[1:]
 assert len(vals)==5
 for val,field in zip(vals,['AP','AP50','AP75','AR500','class_macro_AP_separate_diagnostic']):assert float(val)==round(merged[key][field],3)
for path in (ROOT/'tables').glob('*.tex'):
 assert 'FPS' not in path.read_text() and 'img/s' not in path.read_text(),path.name
print('PASS: original VAL accuracy retained; 18/19/20 identities, metrics, CIs and three new tables; FPS absent from v12 tables.')
