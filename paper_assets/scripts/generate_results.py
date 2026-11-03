"""Regenerate paper tables and plots from the frozen evidence master."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / 'evidence/results.json').read_text(encoding='utf-8'))
M = D['metrics']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42, 'savefig.bbox': 'tight'})


def write(name, text):
    (ROOT / 'tables' / name).write_text(text, encoding='utf-8')


def number(x):
    return f'{x:.3f}'


accuracy = [('YOLOv9-C', 'baseline'), ('Full-image RT-DETR-L', 'rtdetr_full960'),
            ('Global fusion, $\\alpha=1$', 'full_fusion_a100'),
            ('Global fusion, $\\alpha=0.25$', 'full_fusion_a025'),
            ('Fixed grid + YOLO', 'fixed_grid_a025'), ('Relocation mean (3 placements)', 'relocated_mean'),
            ('HyCADO', 'hybrid_a025')]
text = '\\begin{tabular}{lrrrrr}\n\\toprule\nMethod & $\\APlit$ & AP50 & AP75 & AR500 & Macro AP\\\\\n\\midrule\n'
for label, key in accuracy:
    values = [number(M[key][k]) for k in ['AP', 'AP50', 'AP75', 'AR500', 'class_macro_AP_separate_diagnostic']]
    if key == 'hybrid_a025':
        label = '\\textbf{' + label + '}'
        values = ['\\textbf{' + v + '}' for v in values]
    text += label + ' & ' + ' & '.join(values) + '\\\\\n'
write('accuracy.tex', text + '\\bottomrule\n\\end{tabular}\n')

comparisons = [('YOLO', 'hybrid_minus_baseline'), ('Relocation mean', 'hybrid_minus_relocated_mean'),
               ('Fixed grid', 'hybrid_minus_fixed_grid_a025')]
text = '\\begin{tabular}{lrr}\n\\toprule\nHyCADO minus & AP difference & 95\\% interval\\\\\n\\midrule\n'
for label, key in comparisons:
    v = D['CI_cluster'][key]['AP']
    text += f"{label} & +{v['point']:.3f} & [{v['lower95']:.3f}, {v['upper95']:.3f}]\\\\\n"
write('paired_ci.tex', text + '\\bottomrule\n\\end{tabular}\n')

text = '\\begin{tabular}{lrrrr}\n\\toprule\nMethod & Mean (ms) & P95 (ms) & img/s & Alloc. MiB\\\\\n\\midrule\n'
for label, key in [('YOLO', 'baseline'), ('Full RT-DETR', 'rtdetr_full960'), ('HyCADO', 'hybrid_a025')]:
    v = D['runtime_common']['variants'][key]
    text += f"{label} & {v['mean_ms']:.3f} & {v['p95_ms']:.3f} & {v['throughput_images_per_second']:.3f} & {v['peak_allocated_mib']:.3f}\\\\\n"
write('runtime_common.tex', text + '\\bottomrule\n\\end{tabular}\n')

scalars = {'APYolo': M['baseline']['AP'], 'APHybrid': M['hybrid_a025']['AP'],
           'APGlobal': M['full_fusion_a025']['AP'], 'APGlobalRaw': M['full_fusion_a100']['AP'],
           'GainYolo': M['hybrid_a025']['AP'] - M['baseline']['AP'],
           'GainLocation': M['hybrid_a025']['AP'] - M['relocated_mean']['AP'],
           'GainGlobal': M['hybrid_a025']['AP'] - M['full_fusion_a025']['AP'],
           'HybridFPS': D['runtime_common']['variants']['hybrid_a025']['throughput_images_per_second'],
           'HybridLatency': D['runtime_common']['variants']['hybrid_a025']['mean_ms'],
           'MeanCrops': D['location_control']['crops_per_seed'] / 548}
write('scalars.tex', ''.join('\\newcommand{\\' + k + '}{' + number(v) + '}\n' for k, v in scalars.items()))

fig, ax = plt.subplots(figsize=(6.8, 2.55))
colors = ['#225d83', '#c66b24', '#3e7c69']
for y, ((label, key), color) in enumerate(zip(comparisons, colors)):
    v = D['CI_cluster'][key]['AP']
    ax.plot([v['lower95'], v['upper95']], [y, y], color=color, lw=2)
    ax.scatter([v['point']], [y], s=45, color=color, zorder=3)
    ax.text(2.57, y, f"{v['point']:.3f} [{v['lower95']:.3f}, {v['upper95']:.3f}]", va='center', fontsize=8)
ax.set_yticks(range(3), ['vs YOLO', 'vs relocation mean', 'vs fixed grid'])
ax.invert_yaxis(); ax.set_ylim(2.55, -.55); ax.set_xlim(-.08, 3.62)
ax.axvline(0, color='#69737b', ls=':', lw=1)
ax.set_xlabel(r'Paired $\\Delta$AP$_{\\mathrm{toolkit}}$ (points)'.replace('\\\\','\\'))
ax.set_xticks([0, .5, 1, 1.5, 2, 2.5]); ax.grid(axis='x', color='#e1e6ea', lw=.6)
ax.set_axisbelow(True); fig.tight_layout()
fig.savefig(ROOT / 'figures/paired_differences.pdf'); fig.savefig(ROOT / 'figures/paired_differences.png', dpi=300)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.7), gridspec_kw={'width_ratios': [1.2, 1]})
for ax, title, rows, data, limits in [
    (axs[0], '(a) Common session: 3 passes', [('YOLO', 'baseline'), ('Full RT-DETR', 'rtdetr_full960'), ('HyCADO', 'hybrid_a025')], D['runtime_common']['variants'], (20, 1800)),
    (axs[1], '(b) Grid session: 2 passes', [('Fixed grid', 'fixed_grid_a025'), ('HyCADO', 'hybrid_a025')], D['runtime_grid']['variants'], (500, 1500))]:
    for y, (label, key) in enumerate(rows):
        value = data[key]['mean_ms']; color = '#c66b24' if key == 'hybrid_a025' else '#225d83'
        ax.scatter(value, y, s=45, color=color, zorder=3)
        ax.annotate(f'{value:.1f}', (value, y), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=8)
    ax.set_xscale('log'); ax.set_xlim(*limits); ax.set_ylim(len(rows)-.5, -.6)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows]); ax.set_title(title, fontsize=9, pad=15)
    ax.set_xlabel('Mean ms/image (log scale)'); ax.grid(axis='x', color='#e1e6ea', which='both', lw=.5); ax.set_axisbelow(True)
fig.tight_layout(w_pad=2)
fig.savefig(ROOT / 'figures/latency_sessions.pdf'); fig.savefig(ROOT / 'figures/latency_sessions.png', dpi=300)
plt.close(fig)
print('Tables and two scientific figures regenerated from evidence/results.json')
