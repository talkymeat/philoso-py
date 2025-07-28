import unittest
from test_materials import paramarama, shhhh
from philoso_py import ModelFactory
import logging
import sys
from collections import defaultdict
from guardrails import *
import numpy as np
from m import MTuple as T

def fmtp(b: bool)->int:
    return (b*2)-1

def fit_to_int_range_factory(min, max):
    def fit_to_int_range(x):
        x = np.tanh(x)
        x += 1
        x *= (max-min)/2.0
        x += min
        return np.int32(x)
    return fit_to_int_range

class TestGuardrails(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.log = logging.getLogger( "TestGuardrails" )

    def test_tanh_arg_extremum(self):
        for type_, max_ in T([np.float16, np.float32, np.float64]) * T([True, False]):
            self.assertEqual(fmtp(max_), np.tanh(tanh_arg_extremum(type_, max_)))
            self.assertNotEqual(fmtp(max_), np.tanh(nextafter(tanh_arg_extremum(type_, max_), 0)))
            # self.log.debug(f'{tanh_arg_extremum(type_, max_)}')

    def test_exp_arg_extremum(self):
        for type_, max_ in T([np.float16, np.float32, np.float64]) * T([True, False]):
            self.assertEqual(np.inf if max_ else 0.0, np.exp(exp_arg_extremum(type_, max_)))
            self.assertNotEqual(np.inf if max_ else 0.0, np.exp(nextafter(exp_arg_extremum(type_, max_), 0)))
            # self.log.debug(f'{exp_arg_extremum(type_, max_)}')

    def test_tanh_only_guardrail(self):
        tanh_only = TanhGuardrail(dtype=np.float64)
        tanh_min = tanh_arg_extremum(np.float64, max_=False)
        tanh_max = tanh_arg_extremum(np.float64, max_=True)
        eta = 0.0000000001
        self.assertEqual(str(tanh_only.interval), "[-1.0, 1.0]")
        self.assertEqual(i(tanh_only(tanh_min, -1.0)), -1.0)
        self.assertEqual(i(tanh_only.gm.reward), 0.0)
        self.assertEqual(i(tanh_only(tanh_max, 1.0)), 1.0)
        self.assertEqual(i(tanh_only.gm.reward), 0.0)
        self.assertEqual(i(tanh_only(tanh_min-eta, -1.0)), -1.0)
        self.assertLess(i(abs(tanh_only.gm.reward - (-(np.log(1.0+eta)+1.0)))), eta*1e-4)
        self.assertEqual(i(tanh_only.gm.reward), 0.0)
        self.assertEqual(i(tanh_only(tanh_max+eta, 1.0)), 1.0)
        self.assertLess(i(abs(tanh_only.gm.reward - (-1.0-eta))), eta*1e-4)
        self.assertEqual(i(tanh_only.gm.reward), 0.0)

    def test_truncated_tanh_guardrail(self):
        trunc = TanhGuardrail(2.0, 100.0, dtype=np.float64)
        tanh_min = tanh_arg_extremum(np.float64, max_=False)
        eta = 0.0000000001
        f = fit_to_int_range_factory(0, 102)
        self.assertEqual(i(trunc(tanh_min-eta, -1.0)), 2.0)
        self.assertLess(i(abs(trunc.gm.reward - (-1.0-eta))), eta*1e-4)
        self.assertEqual(trunc(-7.0, f(-7.0)), 2.0)
        self.assertEqual(i(trunc.gm.reward), -1.0)
        self.assertEqual(trunc(tanh_min, 0.0), 2.0)
        self.assertLess(i(abs(trunc.gm.reward - (-(np.log(-tanh_min-6.0)+1.0)))), eta*1e-4)
        self.assertEqual(trunc(-2.5, f(-2.5)), 2.0)
        self.assertEqual(i(trunc.gm.reward), -1.0)
        self.assertEqual(trunc(-7.0, f(-7.0)), 2.0)
        self.assertLess(i(abs(trunc.gm.reward - -(np.log(5.5)+1.0))), eta*1e-4)
        self.assertEqual(trunc(-2.0, f(-2.0)), 2.0)
        self.assertEqual(i(trunc.gm.reward), -1.0)
        self.assertEqual(trunc(-2.5, f(-2.5)), 2.0)
        self.assertLess(i(abs(trunc.gm.reward - -(np.log(1.5)+1.0))), eta*1e-4)
        self.assertEqual(i(trunc(-7.0, f(-7.0))), 2.0)
        self.assertLess(i(abs(trunc.gm.reward - -(np.log(6.0)+1.0))), eta*1e-4)
        self.assertEqual(i(trunc(-1.9, f(-1.9))), 2)
        self.assertEqual(i(trunc.gm.reward), 0.0)
        self.assertEqual(i(trunc(-1.0, 12)), 12)
        self.assertEqual(i(trunc.gm.reward), 0.0)
        self.assertEqual(trunc(2.5, 101.0), 100.0)
        self.assertEqual(i(trunc.gm.reward), -1.0)
        self.assertEqual(i(trunc(19, 102.0)), 100.0)
        self.assertLess(i(abs(trunc.gm.reward - -(np.log(17.5)+1.0))), eta*1e-4)

    def test_exp_only_guardrail(self):
        exp_only = ExponentialGuardrail(dtype=np.float64)
        exp_min = exp_arg_extremum(np.float64, max_=False)
        exp_max = exp_arg_extremum(np.float64, max_=True)
        self.assertFalse(exp_only.interval.closed_hi)
        self.assertEqual(exp_only(exp_max, np.inf), np.nextafter(np.inf, 0.0))
        self.assertEqual(exp_only(exp_min, 0.0), 0.0)
        self.assertAlmostEqual(exp_only.gm.reward, -1.0)
        self.assertEqual(exp_only.gm.reward, 0.0)
        self.assertEqual(exp_only(exp_min+1.0-np.e**3, 0.0), 0.0)
        self.assertAlmostEqual(exp_only.gm.reward, -4.0)
        self.assertNotEqual(exp_only(exp_max-1.0+np.e**3, np.inf), np.inf)
        self.assertAlmostEqual(exp_only.gm.reward, -4.0)
        self.assertEqual(exp_only(np.log(1e12), 1e12), 1e12)

    def test_exp_reversed_guardrail(self):
        exp_only = ExponentialGuardrail(-np.inf, 0.0, False, True, reversed_=True, dtype=np.float64)
        exp_min = exp_arg_extremum(np.float64, max_=False)
        exp_max = exp_arg_extremum(np.float64, max_=True)
        self.assertFalse(exp_only.interval.closed_lo)
        self.assertEqual(exp_only(exp_max, -np.inf), np.nextafter(-np.inf, 0.0))
        self.assertEqual(exp_only(exp_min, 0.0), 0.0)
        self.assertAlmostEqual(exp_only.gm.reward, -1.0)
        self.assertEqual(exp_only.gm.reward, 0.0)
        self.assertEqual(exp_only(exp_min+1.0-np.e**3, 0.0), 0.0)
        self.assertAlmostEqual(exp_only.gm.reward, -4.0)
        self.assertNotEqual(exp_only(exp_max-1.0+np.e**3, -np.inf), -np.inf)
        self.assertAlmostEqual(exp_only.gm.reward, -4.0)
        self.assertEqual(exp_only(np.log(1e12), -1e12), -1e12)

    def test_exp_trunc_guardrail(self):
        exp_trunc = ExponentialGuardrail(np.exp(-4.0), np.exp(4.0), dtype=np.float64)
        exp_min = exp_arg_extremum(np.float64, max_=False)
        exp_max = exp_arg_extremum(np.float64, max_=True)
        self.assertEqual(exp_trunc(0.0, 1.0), 1.0)
        self.assertEqual(exp_trunc.gm.reward, 0.0)
        self.assertEqual(exp_trunc(-4.0, np.exp(-4.0)), np.exp(-4.0))
        self.assertEqual(exp_trunc.gm.reward, 0.0)
        self.assertEqual(exp_trunc(4.0, np.exp(4.0)), np.exp(4.0))
        self.assertEqual(exp_trunc.gm.reward, 0.0)
        self.assertEqual(exp_trunc.raw_interval, Interval(exp_min, exp_max))
        self.assertEqual(exp_trunc(-5.0, np.exp(-5.0)), np.exp(-4.0))
        self.assertEqual(exp_trunc.raw_interval, Interval(-5.0, exp_max))
        self.assertEqual(exp_trunc.gm.reward, -1.0)
        self.assertEqual(exp_trunc(-6.0, np.exp(-6.0)), np.exp(-4.0))
        self.assertEqual(exp_trunc.raw_interval, Interval(-5.0, exp_max))
        self.assertAlmostEqual(exp_trunc.gm.reward, -(1.0+np.log(2.0)))
        self.assertEqual(exp_trunc(5.0, np.exp(5.0)), np.exp(4.0))
        self.assertEqual(exp_trunc.raw_interval, Interval(-5.0, 5.0))
        self.assertEqual(exp_trunc.gm.reward, -1.0)
        self.assertEqual(exp_trunc(6.0, np.exp(6.0)), np.exp(4.0))
        self.assertEqual(exp_trunc.raw_interval, Interval(-5.0, 5.0))
        self.assertEqual(exp_trunc.gm.reward, -(1.0+np.log(2.0)))


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestGuardrails").setLevel(logging.DEBUG)
    unittest.main()