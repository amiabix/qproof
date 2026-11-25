import unittest

import numpy as np
import m0_certificate_reference as cert


class WeightScaleTest(unittest.TestCase):
    def test_rejects_out_of_range_scale_before_integer_shift(self):
        weights = np.zeros((1, 2560))
        weights[0, :2] = [1, 2**-64]
        inputs = cert.bf16(np.ones_like(weights))
        with self.assertRaisesRegex(ValueError, "weight scale exceeds encoding bound"):
            cert.encode(cert.bf16(weights), inputs, (np.array([80]), np.array([75])))
