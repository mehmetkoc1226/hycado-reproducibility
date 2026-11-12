"""Regenerate eleven v12 LaTeX files from recorded results, using the standard library."""
from pathlib import Path
import json
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'evidence'

def read(name):
    return json.loads((E / name).read_bytes())

def write(name, content):
    (ROOT / 'tables' / name).write_text(content, encoding='utf-8')

def table(columns, header, rows):
    end = '\\\\\n'
    return ('\\begin{tabular}{' + columns + '}\n\\toprule\n' + header + end
            + '\\midrule\n' + ''.join(' & '.join(row) + end if row is not None
                                      else '\\midrule\n' for row in rows)
            + '\\bottomrule\n\\end{tabular}\n')

def main():
    d = read('results.json')
    m = d['metrics'] | read('p2_val/official_val_metrics.json')['methods']
    conditions = [
        ('YOLOv9-C at 800 (YOLO-800)', 'baseline'),
        ('YOLOv9-C at 1280 (YOLO-1280)', 'yolo1280'),
        ('YOLOv9-C at 1536 (YOLO-1536)', 'yolo1536'),
        ('YOLOv9-C + SAHI-640', 'sahi640'), ('YOLOv9-C + SAHI-800', 'sahi800'),
        ('HyCADO v1 (YOLO-800 + crop RT-DETR-L)', 'hybrid_a025'), (None, None),
        ('Full-image RT-DETR-L', 'rtdetr_full960'),
        (r'Global fusion, $\alpha=1$', 'full_fusion_a100'),
        (r'Global fusion, $\alpha=0.25$', 'full_fusion_a025'),
        ('Fixed grid + YOLO-800', 'fixed_grid_a025'),
        ('Relocation mean (three placements)', 'relocated_mean')]
    fields = ['AP', 'AP50', 'AP75', 'AR500', 'class_macro_AP_separate_diagnostic']
    rows = [None if key is None else [label, *[f'{m[key][k]:.3f}' for k in fields]]
            for label, key in conditions]
    write('accuracy.tex', table('lrrrrr', r'Method & $\APlit$ & AP50 & AP75 & AR500 & Macro AP', rows))
    rows = []
    global_ci = read('global_intervals/intervals.json')['cluster']
    strong_ci = read('p2_intervals/intervals.json')['bootstrap_modes']['cluster']
    for label, ci in [('YOLO-800', d['CI_cluster']['hybrid_minus_baseline']),
                      ('Relocation mean', d['CI_cluster']['hybrid_minus_relocated_mean']),
                      ('Fixed grid', d['CI_cluster']['hybrid_minus_fixed_grid_a025']),
                      ('Global fusion (0.25)', global_ci['hybrid_minus_full_fusion_a025'])]:
        v = ci['AP']
        rows.append([label, f"{v['point']:+.3f}",
                     f"[{v['lower95']:.3f}, {v['upper95']:.3f}]", '2000'])
    for label, key in [('YOLO-1280', 'yolo1280'), ('YOLO-1536', 'yolo1536'),
                       ('SAHI-640', 'sahi640'), ('SAHI-800', 'sahi800')]:
        v = strong_ci['contrasts_hycado_minus_baseline'][key]['AP']
        rows.append([label, f"{v['point']:+.3f}",
                     f"[{v['lower95']:.3f}, {v['upper95']:.3f}]", '1000'])
    write('paired_ci.tex', table('lrrr', r'HyCADO v1 minus & $\Delta$AP & 95\% interval & Draws', rows))
    scalars = {'APYolo': m['baseline']['AP'], 'APHybrid': m['hybrid_a025']['AP'],
               'APGlobal': m['full_fusion_a025']['AP'], 'APGlobalRaw': m['full_fusion_a100']['AP'],
               'GainYolo': m['hybrid_a025']['AP'] - m['baseline']['AP'],
               'GainLocation': m['hybrid_a025']['AP'] - m['relocated_mean']['AP'],
               'GainGlobal': m['hybrid_a025']['AP'] - m['full_fusion_a025']['AP'],
               'HybridLatency': read('cost_disclosure.json')['original_hycado800_mean_ms'],
               'MeanCrops': d['location_control']['crops_per_seed'] / 548}
    strong = {'APYoloHigh': m['yolo1280']['AP'],
              'HighAPLead': m['yolo1280']['AP'] - m['hybrid_a025']['AP'],
              'ReplayAPGain': read('old_crop_replay/result.json')['AP_gain_vs_yolo1280_pp']}
    for name, values in [('scalars.tex', scalars), ('strong_scalars.tex', strong)]:
        write(name, ''.join('\\newcommand{\\' + k + '}{' + f'{v:.3f}' + '}\n'
                            for k, v in values.items()))
    for script in ['check_extensions.py', 'check_val_diagnostics.py']:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / script), '--write-tables'], check=True)
    print('Generated eleven v12 LaTeX files; no inference or AP recomputation.')

if __name__ == '__main__':
    main()
