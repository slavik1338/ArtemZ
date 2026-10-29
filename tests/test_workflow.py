import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from libs import analyze, read_spectrum, train


class WorkflowTest(unittest.TestCase):
    def test_training_and_identification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            lines = root / 'lines.csv'
            # Deliberately fictional lines: these are not NIST reference data.
            lines.write_text('element,ion,wavelength_nm,relative_strength\n'
                             'A,I,501,1\nA,I,502,0.7\nB,I,504,1\nB,I,505,0.8\n')
            x = np.linspace(500, 506, 1201)
            y = .2 + np.exp(-.5*((x-501)/.025)**2)
            y += .7*np.exp(-.5*((x-502)/.025)**2)
            spectrum = root / 'spectrum.csv'
            with spectrum.open('w') as f:
                writer = csv.writer(f)
                writer.writerow(['wavelength_nm', 'intensity'])
                writer.writerows(zip(x, y))
            model = root / 'model.joblib'
            report = train(spectrum, lines, model, count=500, tolerance=.06)
            self.assertGreater(report['synthetic_micro_f1'], .8)
            result = analyze(spectrum, model)
            scores = {e['element']: e['synthetic_model_score'] for e in result['elements']}
            self.assertGreater(scores['A'], scores['B'])
            self.assertEqual(len(result['matched_peaks']), 2)
            self.assertTrue(all(e['concentration'] is None for e in result['elements']))

    def test_duplicate_wavelengths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.csv'
            path.write_text('wavelength_nm,intensity\n' + '500,1\n'*20)
            with self.assertRaises(ValueError):
                read_spectrum(path)


if __name__ == '__main__':
    unittest.main()
