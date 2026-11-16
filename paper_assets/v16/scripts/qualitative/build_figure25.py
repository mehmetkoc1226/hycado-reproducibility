"""Publication figures from the byte-verified Stage25 export. No new inference.

Usage: python build_figure25.py --input evidence/25_nitel_gorsel_rapor.zip --out figures
Requires Pillow, NumPy, ReportLab; pdftoppm is used separately to rasterize PDFs.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile
from PIL import Image
import reportlab
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader, simpleSplit
from reportlab.lib.pagesizes import A4

FONT = 'FigureSans'
BOLD = 'FigureSansBold'
FONT_DIR = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont(FONT, str(FONT_DIR / 'Vera.ttf')))
pdfmetrics.registerFont(TTFont(BOLD, str(FONT_DIR / 'VeraBd.ttf')))

METHODS = ('ground_truth', 'yolo1280', 'hybrid1280', 'sahi640')
TITLES = ('Ground truth', 'YOLO-1280', 'HyCADO-1280', 'SAHI-640')
NAMES = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
         'tricycle', 'awning-tricycle', 'bus', 'motor')
COLORS = ('#D62728', '#E377C2', '#9467BD', '#1F77B4', '#17BECF',
          '#FF7F0E', '#BCBD22', '#8C564B', '#2CA02C', '#F2B701')
EXPECTED_EXPORT_SHA256 = '4f211056558dfe44fcad96a6f3ca29a89edafa544b99d7d5b02bc45a1d68fe11'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(source, target):
    assert sha(Path(source).read_bytes()) == EXPECTED_EXPORT_SHA256, 'A different Stage25 export was supplied'
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        hashes = json.loads(z.read('contents_sha256.json'))
        assert set(z.namelist()) == set(hashes) | {'contents_sha256.json'}
        for name in z.namelist():
            path = target / name
            assert path.resolve().is_relative_to(target.resolve()), 'Unsafe ZIP member'
            content = z.read(name)
            if name in hashes:
                assert sha(content) == hashes[name], 'File digest differs: ' + name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)


def prepare(source):
    examples = read(source / 'examples.json')
    selection = read(source / 'selection.json')
    protocol = read(source / 'protocol.json')
    assert read(source / 'result.json')['status'] == 'QUALITATIVE_EXPORT_COMPLETE'
    assert len(examples) == len(selection) == 12
    assert len({x['group'] for x in examples}) == 12
    # Import only the pinned helper distributed alongside this renderer.
    # No code from the incoming ZIP is executed.
    import qualitative25 as q
    assert (source / 'qualitative25.py').read_bytes() == Path(q.__file__).read_bytes()
    answer = []
    for item, expected in zip(examples, selection):
        assert all(item[k] == v for k, v in expected.items())
        rec = item['record']
        stem = rec['stem']
        raw_image = (source / 'images' / (stem + '.jpg')).read_bytes()
        annotation = (source / 'annotations' / rec['annotation']).read_bytes()
        assert sha(raw_image) == rec['image_sha256']
        assert sha(annotation) == item['annotation_sha256']
        bodies = {}
        for method in METHODS[1:]:
            raw = (source / 'predictions' / (stem + '__' + method + '.json')).read_bytes()
            assert sha(raw) == item['prediction_sha256'][method]
            bodies[method] = json.loads(raw)
        assert q.illustration_data(annotation, bodies, rec, protocol['qualitative']) == item['illustration']
        with Image.open(io.BytesIO(raw_image)) as image:
            assert image.size == (rec['width'], rec['height']) and image.format == 'JPEG'
            full = image.convert('RGB')
        window = item['illustration']['window_xyxy']
        answer.append((item, full, full.crop(tuple(window)), raw_image))
    assert [x[0]['rank'] for x in answer[:3]] == [1, 2, 3]
    assert all(x[0]['role'] == 'main' for x in answer[:3])
    return answer


def label(c, text, x, y, size=8, bold=False, center=False):
    c.setFillColor(HexColor('#18222D'))
    c.setFont(BOLD if bold else FONT, size)
    if center:
        c.drawCentredString(x, y, text)
    else:
        c.drawString(x, y, text)


def panel(c, item, crop, method, x, y, width, linewidth=.52):
    window = item['illustration']['window_xyxy']
    left, top, right, bottom = window
    scale = width / (right - left)
    height = (bottom - top) * scale
    c.saveState()
    path = c.beginPath()
    path.rect(x, y, width, height)
    c.clipPath(path, stroke=0, fill=0)
    # The same native RGB crop is reused without brightness/contrast edits.
    c.drawImage(ImageReader(crop), x, y, width=width, height=height, mask=None)
    c.setLineWidth(linewidth)
    c.setLineJoin(0)
    for row in item['illustration']['detections_xyxy_score_class0'][method]:
        a, b, cc, d = max(left, row[0]), max(top, row[1]), min(right, row[2]), min(bottom, row[3])
        if a >= cc or b >= d:
            continue
        c.setStrokeColor(HexColor(COLORS[int(row[5])]))
        c.rect(x + (a - left) * scale, y + (bottom - d) * scale,
               (cc - a) * scale, (d - b) * scale, stroke=1, fill=0)
    c.restoreState()


def legend(c, x, y, width, fontsize=6.1, spacing=11):
    cell = width / 5
    for i, (name, color) in enumerate(zip(NAMES, COLORS)):
        xx = x + (i % 5) * cell
        yy = y - (i // 5) * spacing
        c.setFillColor(HexColor(color))
        c.rect(xx, yy - .5, 5, 5, fill=1, stroke=0)
        label(c, name, xx + 8, yy, fontsize)


def main_figure(examples, output):
    width, left, gap, header, rowgap, bottom = 518.4, 13., 5., 18., 10., 31.
    pw = (width - left - 3 * gap) / 4
    ph = pw * 420 / 640
    height = header + 3 * ph + 2 * rowgap + bottom
    c = canvas.Canvas(str(output), pagesize=(width, height), pageCompression=1)
    c.setTitle('HyCADO final-configuration qualitative comparison')
    c.setAuthor('Mehmet Koc')
    for j, title in enumerate(TITLES):
        label(c, title, left + j * (pw + gap) + pw / 2, height - 10., 8., True, True)
    for i, (item, full, crop, original_jpeg) in enumerate(examples[:3]):
        y = height - header - ph - i * (ph + rowgap)
        label(c, '(' + chr(97 + i) + ')', .5, y + ph - 8., 7.5, True)
        for j, method in enumerate(METHODS):
            panel(c, item, crop, method, left + j * (pw + gap), y, pw)
    legend(c, left, 17., width - left, fontsize=6.1)
    c.showPage()
    c.save()


def supplementary(examples, output):
    width, height = A4
    margin, gap = 22., 12.
    pw = (width - 2 * margin - gap) / 2
    ph = pw * 420 / 640
    c = canvas.Canvas(str(output), pagesize=A4, pageCompression=1)
    c.setTitle('HyCADO fixed twelve-image qualitative sample')
    c.setAuthor('Mehmet Koc')
    for item, full, crop, original_jpeg in examples:
        rec = item['record']
        label(c, 'Fixed-sample qualitative comparisons', margin, height - 27, 12., True)
        label(c, f"{item['rank']:02d}/12  {rec['stem']}", margin, height - 44, 8.5)
        scale = min(240 / full.width, 150 / full.height)
        fw, fh = full.width * scale, full.height * scale
        fx, fy = margin, height - 62 - fh
        # Embed the original JPEG directly: no decode/recompression for context.
        c.drawImage(ImageReader(io.BytesIO(original_jpeg)), fx, fy, width=fw, height=fh)
        x1, y1, x2, y2 = item['illustration']['window_xyxy']
        c.setStrokeColor(HexColor('#FFE000'))
        c.setLineWidth(1.25)
        c.rect(fx + x1 * scale, fy + (full.height - y2) * scale,
               (x2 - x1) * scale, (y2 - y1) * scale, stroke=1, fill=0)
        info = [f"Role: {'main figure' if item['role'] == 'main' else 'additional fixed-sample example'}",
                f"Filename-based proxy group: {item['group']}",
                f"Original image: {full.width} x {full.height} pixels",
                f"Window (x1,y1,x2,y2): {x1}, {y1}, {x2}, {y2}",
                'Native window: 640 x 420 pixels',
                'Displayed predictions: archived score >= 0.05',
                'Colors: object classes; no TP/FP labels',
                'Window selected from GT; fixed image sample',
                'Yellow rectangle: window location']
        for i, text in enumerate(info):
            label(c, text, 278., height - 73 - i * 13, 7.6)
        for j, (method, title) in enumerate(zip(METHODS, TITLES)):
            col, row = j % 2, j // 2
            x = margin + col * (pw + gap)
            top = 598. - row * (ph + 31.)
            label(c, title, x + pw / 2, top + 9., 9., True, True)
            panel(c, item, crop, method, x, top - ph, pw, .62)
        legend(c, margin, 184., width - 2 * margin, 6.6, 12.)
        paragraphs = [
            'The twelve images were fixed by SHA256 ranking with distinct filename-based proxy groups; the first three appear in the main figure. Image selection does not use detection performance.',
            'All methods use the same GT-selected native-resolution window and pinned ignore-region preprocessing. The common numerical display threshold does not match false-positive rates or recalibrate the different models. No new inference, matching, or accuracy measurement is performed.',
            'Source: VisDrone DET test-dev. Original images, annotations, frozen predictions, exact windows, and provenance hashes accompany the figure package.'
        ]
        y = 148.
        for text in paragraphs:
            for line in simpleSplit(text, FONT, 8., width - 2 * margin):
                label(c, line, margin, y, 8.)
                y -= 11.
            y -= 6.
        label(c, 'Illustrative examples; quantitative conclusions come from the full evaluation.',
              margin, 25., 7.)
        label(c, str(item['rank']), width - margin, 25., 7.)
        c.showPage()
    c.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='hycado25_render_') as temporary:
        source = Path(temporary)
        extract(args.input, source)
        examples = prepare(source)
        main_figure(examples, args.out / 'qualitative1280_main.pdf')
        supplementary(examples, args.out / 'qualitative1280_supplementary.pdf')
    print('Rendered three fixed main examples and all twelve supplementary examples.')


if __name__ == '__main__':
    main()
