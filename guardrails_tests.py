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

class TestGuardrails(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.log = logging.getLogger( "TestGuardrails" )

    def test_tanh_arg_extremum(self):
        for type_, max_ in T([np.float16, np.float32, np.float64, np.longdouble]) * T([True, False]):
            self.assertEqual(fmtp(max_), np.tanh(tanh_arg_extremum(type_, max_)))
            self.assertNotEqual(fmtp(max_), np.tanh(nextafter(tanh_arg_extremum(type_, max_), 0)))
            self.log.debug(f'{tanh_arg_extremum(type_, max_)}')

    def test_exp_arg_extremum(self):
        for type_, max_ in T([np.float16, np.float32, np.float64]) * T([True, False]):
            self.assertEqual(np.inf if max_ else 0.0, np.exp(exp_arg_extremum(type_, max_)))
            self.assertNotEqual(np.inf if max_ else 0.0, np.exp(nextafter(exp_arg_extremum(type_, max_), 0)))
            self.log.debug(f'{exp_arg_extremum(type_, max_)}')


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr)
    logging.getLogger("TestGuardrails").setLevel(logging.DEBUG)
    unittest.main()