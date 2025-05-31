import unittest
import numpy as np
from icecream import ic

from mutators import *
from test_materials import Dummy
from utils import aeq

mt=ModelTime()

class TestMut8ors(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)