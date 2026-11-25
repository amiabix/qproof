"""Small synthetic checks for the pinned-dot certificate."""

import unittest
from fractions import Fraction as F

import numpy as np

if __package__:
    from .certificate import assess, lower_boundary
else:
    from certificate import assess, lower_boundary


def row(*values, k=32):
    a = np.zeros((1, k), dtype=np.float32)
    a[0, :len(values)] = values
    return a


class CertificateTests(unittest.TestCase):
    def test_signed_midpoints(self):
        step = F(1, 2**133)
        for y, expected in (
            (0., -step/2), (-0., -step/2),
            (step, step/2), (-step, -3*step/2),
            (127*step, 253*step/2), (128*step, 255*step/2),
            (-128*step, -257*step/2), (256*step, 511*step/2),
            (1, 1-F(1, 512)), (-1, -1-F(1, 256)),
            (F(129, 128), F(257, 256)),
            (-255 * 2**120, -511 * 2**119),
        ):
            with self.subTest(y=y):
                boundary = lower_boundary(float(y))
                self.assertIsInstance(boundary, F)
                self.assertEqual(boundary, expected)

    def test_exact_truncation_terms_and_running_maxima(self):
        maxima = {}
        w, x = row(-1, 2.**-80), row(1, 2.**-80)
        result, = assess(w, x, maxima)
        bs = 5 * (2**32 + 1)
        expected = dict(S=2**45, Bs=bs, E1=1, E2=1, E3=1)
        self.assertEqual(maxima, expected)
        total = (F(bs, 2**32 * (2**24 - 5)) + F(1, 2**37)
                 + F(1, 2**40) + F(1, 2**45) + F(64, 2**150))
        self.assertEqual(result, (0, 0, "certified", -1., -1 + total))
        self.assertEqual(list(assess(row(-0.), row(0.), maxima)),
                         [(0, 0, "E3", None, F(64, 2**150))])
        self.assertEqual(maxima, expected)

    def test_scales_and_gates(self):
        for w, x, status, y in (
            (row(2.**80), row(-2.**-72), "certified", -256.),
            (row(2.**40), row(2.**40), "certified", 2.**80),
            (row(2.**126), row(2.), "E2", None),
            (row(2.**-126), row(1.), "E3", None),
            (row(2.**-133), row(1.), "E3", None),
            (row(1., -1.), row(1., 1.), "E3", None),
            (row(-1., -2.**-8), row(1., 1.), "E4", None),
            (row(1., -2.**-9), row(1., 1.), "E4", None),
        ):
            with self.subTest(status=status, w=w[0, 0]):
                (r, t, actual, value, upper), = assess(w, x)
                self.assertEqual((r, t, actual, value), (0, 0, status, y))
                if status == "E2":
                    self.assertIsNone(upper)
                else:
                    self.assertIsInstance(upper, F)
                    exact = sum(F(float(a))*F(float(b)) for a, b in zip(w[0], x[0]))
                    self.assertGreaterEqual(upper, exact)

    def test_supported_width_and_field_bounds(self):
        k = 9728
        ones = np.ones((1, k), np.float32)
        maxima = {}
        (_, _, status, y, _), = assess(-ones, ones, maxima)
        self.assertEqual((status, y), ("certified", -float(k)))
        self.assertEqual(maxima['S'], k << 45)
        n = k // 32
        self.assertEqual(maxima['Bs'], 32 * sum(n-max(j, 1)+5 for j in range(n)) << 32)
        self.assertTrue(all(2*v < 2**64 - 2**32 + 1 for v in maxima.values()))
        self.assertLessEqual((k+31)*F(2**24, 2**24-n-4), k+32)
        for width in (0, 31, 33, k+32):
            with self.assertRaises(ValueError):
                list(assess(row(k=width), row(k=width)))

    def test_reject_bad_inputs(self):
        for bad in (np.nan, np.inf, -np.inf, 1 + 2.**-12):
            for pair in ((row(bad), row(1)), (row(1), row(bad))):
                with self.assertRaises(ValueError):
                    list(assess(*pair))
        for bad in (row(1).astype(np.float64), row(1)[0], row(1, k=64)):
            with self.assertRaises(ValueError):
                list(assess(bad, row(1)))

    def test_against_pinned_lane_reduction(self):
        rng = np.random.default_rng(938)
        def inputs(rows):
            mant = rng.integers(-255, 256, (rows, 128)).astype(np.float32)
            return np.ldexp(mant, rng.integers(-20, 4, mant.shape))

        w, x = inputs(6), inputs(4)
        results = list(assess(w, x))
        self.assertEqual([(r, t) for r, t, *_ in results],
                         [(r, t) for r in range(6) for t in range(4)])
        self.assertTrue(any(result[2] == "certified" for result in results))
        for r, t, status, y, upper in results:
            # These BF16 products fit exactly in normal FP32, so add equals FMA.
            lanes = np.zeros(32, dtype=np.float32)
            for group in (w[r] * x[t]).reshape(-1, 32):
                lanes += group
            for offset in (16, 8, 4):
                lanes[:offset] += lanes[offset:2*offset]
            fp = (lanes[0]+lanes[1]) + (lanes[2]+lanes[3])
            self.assertLessEqual(F(float(fp)), upper)
            bits = int(fp.view(np.uint32))
            rounded = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
            if status == "certified":
                self.assertEqual(int(np.float32(y).view(np.uint32)), rounded)


if __name__ == "__main__":
    unittest.main()
