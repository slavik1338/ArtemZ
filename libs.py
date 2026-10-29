"""Experimental LIBS identification; concentrations require independent calibration."""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d, percentile_filter
from scipy.signal import find_peaks
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def read_spectrum(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    x = np.array([float(r['wavelength_nm']) for r in rows])
    y = np.array([float(r['intensity']) for r in rows])
    if len(x) < 20 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Need at least 20 finite wavelength/intensity pairs')
    order = np.argsort(x)
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError('Wavelengths must be unique')
    return x, y


def read_lines(path, lo, hi):
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    result = []
    for r in rows:
        w, strength = float(r['wavelength_nm']), float(r['relative_strength'])
        if not np.isfinite([w, strength]).all() or strength <= 0 or not r['element'].strip():
            raise ValueError('Invalid line wavelength, element or relative_strength')
        if lo <= w <= hi:
            result.append(dict(element=r['element'].strip(), wavelength_nm=w,
                               relative_strength=strength, ion=r.get('ion', '')))
    if not result:
        raise ValueError('No reference lines in the measured range')
    return result


def preprocess(x, y):
    # Resample before filters: input spacing need not be uniform.
    grid = np.linspace(x[0], x[-1], len(x))
    signal = np.interp(grid, x, y)
    window = min(101, max(5, len(x) // 20))
    if window % 2 == 0:
        window += 1
    baseline = percentile_filter(signal, percentile=15, size=window, mode='nearest')
    corrected = np.maximum(gaussian_filter1d(signal - baseline, .7), 0)
    scale = corrected.max()
    if scale <= 0:
        raise ValueError('Spectrum has no signal after background subtraction')
    return grid, corrected / scale


def features(grid, signal, lines, tolerance):
    values = []
    for line in lines:
        mask = np.abs(grid - line['wavelength_nm']) <= tolerance
        values.append(float(signal[mask].max()) if mask.any() else 0.)
    return values


def train(spectrum, line_file, output, count=1000, seed=42, tolerance=.1):
    x, _ = read_spectrum(spectrum)
    if tolerance <= 0 or count < 100:
        raise ValueError('Positive tolerance and at least 100 synthetic spectra required')
    grid = np.linspace(x[0], x[-1], len(x))
    if np.diff(grid)[0] > tolerance:
        raise ValueError('Tolerance is smaller than the resampled wavelength step')
    lines = read_lines(line_file, grid[0], grid[-1])
    elements = sorted({r['element'] for r in lines})
    if len(elements) < 2:
        raise ValueError('At least two candidate elements required')
    rng = np.random.default_rng(seed)
    xx, yy = [], []
    strengths = {e: max(r['relative_strength'] for r in lines if r['element'] == e)
                 for e in elements}
    for _ in range(count):
        present = rng.random(len(elements)) < .5
        amplitudes = rng.uniform(.15, 1., len(elements)) * present
        signal = rng.normal(0, .008, len(grid)) + rng.uniform(0, .1)
        width = rng.uniform(max(np.diff(grid)[0], tolerance / 4), tolerance)
        shift = rng.uniform(-tolerance / 3, tolerance / 3)
        for r in lines:
            a = amplitudes[elements.index(r['element'])]
            a *= r['relative_strength'] / strengths[r['element']]
            a *= rng.lognormal(0, .3)
            signal += a * np.exp(-.5 * ((grid-r['wavelength_nm']-shift)/width)**2)
        gx, sy = preprocess(grid, signal)
        xx.append(features(gx, sy, lines, tolerance))
        yy.append(present.astype(int))
    split = int(count * .8)
    model = make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1200, random_state=seed))
    model.fit(np.array(xx[:split]), np.array(yy[:split]))
    score = f1_score(yy[split:], model.predict(xx[split:]), average='micro', zero_division=0)
    artifact = dict(model=model, lines=lines, elements=elements, grid=grid,
                    tolerance=tolerance, synthetic_micro_f1=float(score), version=1)
    joblib.dump(artifact, output)
    return dict(synthetic_test_spectra=count-split, synthetic_micro_f1=float(score),
                warning='Synthetic validation only; real-world accuracy is unknown.')


def analyze(spectrum, model_path, prominence=.05):
    # joblib uses pickle: load only artifacts you generated or trust.
    artifact = joblib.load(model_path)
    x, y = read_spectrum(spectrum)
    grid = artifact['grid']
    if x[0] > grid[0] or x[-1] < grid[-1]:
        raise ValueError('Spectrum must cover the complete training wavelength range')
    if np.max(np.diff(x)) > artifact['tolerance']:
        raise ValueError('Spectrum sampling is too coarse for model tolerance')
    gx, sy = preprocess(grid, np.interp(grid, x, y))
    peaks, _ = find_peaks(sy, prominence=prominence)
    scores = artifact['model'].predict_proba(np.array([
        features(gx, sy, artifact['lines'], artifact['tolerance'])]))[0]
    matches = []
    for peak in peaks:
        candidates = [dict(r, delta_nm=float(gx[peak]-r['wavelength_nm']))
                      for r in artifact['lines']
                      if abs(gx[peak]-r['wavelength_nm']) <= artifact['tolerance']]
        if candidates:
            matches.append(dict(observed_wavelength_nm=float(gx[peak]),
                                relative_peak_height=float(sy[peak]),
                                ambiguous=len(candidates)>1, candidates=candidates))
    return dict(elements=[dict(element=e, synthetic_model_score=float(p),
                               concentration=None, concentration_unit=None)
                          for e, p in zip(artifact['elements'], scores)],
                matched_peaks=matches,
                warnings=[
                    'Scores are not calibrated probabilities or confirmed detections.',
                    'Concentrations require standards or a separately validated physical model.',
                    'Unmatched elements and lines outside the supplied table cannot be identified.',
                    'Reference and measured wavelengths must use the same air/vacuum convention.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    tr = sub.add_parser('train')
    tr.add_argument('--spectrum', required=True)
    tr.add_argument('--lines', required=True)
    tr.add_argument('--output', required=True)
    tr.add_argument('--count', type=int, default=1000)
    tr.add_argument('--tolerance-nm', type=float, default=.1)
    an = sub.add_parser('analyze')
    an.add_argument('--spectrum', required=True)
    an.add_argument('--model', required=True)
    an.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        result = train(args.spectrum, args.lines, args.output,
                       count=args.count, tolerance=args.tolerance_nm)
    else:
        result = analyze(args.spectrum, args.model)
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
