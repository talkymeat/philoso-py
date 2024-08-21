from icecream import ic
from gp import GPTreebank
import operators as ops
from functools import reduce
from tree_factories import TestTreeFactory, RandomTreeFactory
from gp_poly_test import EXAMPLE_POLY_OBS_FAC as POF
from gp_fitness import SimpleGPScoreboardFactory
from repository import Archive, Publication
import numpy as np
from model_time import ModelTime
from guardrails import GuardrailManager


class DummyAgent:
    def __init__(self, name):
        self.name = name


mt = ModelTime()
AG = [DummyAgent(f"a{id}") for id in range(10)]

MEM = Archive(
    cols=['goodness'], 
    rows=10,
    model_time=mt,
    types={'goodness': np.float64},
    tables=4,
    value='goodness',
)

PLOS1 = Publication( 
    cols=['goodness'], 
    rows=10,
    model_time=mt,
    agent_names={a.name: i for i, a in enumerate(AG)},
    types={'goodness': np.float64},
    tables=3,
    value='goodness',
    reward='ranked'
)
PLOS1.add_users(AG)


class DummyTreeFactory:
    def __init__(self) -> None:
        self.treebank = None

    @property
    def treebank(self)->"GPTreebank":
        return self._treebank

    @treebank.setter
    def treebank(self, treebank: "GPTreebank"):
        self._treebank = treebank

class DummyController:
    def __init__(self) -> None:
        self.gptb_list  = [None] * 4
        self.gptb_cts   = [0] * 4
        self.gp_vars_core = [
            'mse', 'rmse', 'size', 'depth', 'raw_fitness', 'fitness', 'temp_coeff', 
        ]
        self.gp_vars_more = [
            'wt_fitness', 'wt_size', 'wt_depth', "crossover_rate", "mutation_rate", 
            "mutation_sd", "max_depth", "max_size", "temp_coeff", "pop", "elitism", 
            'obs_start', 'obs_stop', 'obs_num'
        ]
        self.gp_vars_out = self.gp_vars_core + self.gp_vars_more
        self.sb_factory = SimpleGPScoreboardFactory(self.gp_vars_core, 'x')
        self.memory = MEM
        self.repository = PLOS1
        self.guardrail_manager = GuardrailManager()

DTF = DummyTreeFactory()
DC = DummyController()
GP = GPTreebank(operators=[ops.SUM, ops.PROD, ops.SQ, ops.CUBE, ops.POW], tree_factory=DummyTreeFactory())
T0 = GP.tree('([float]<SUM>([float]<SQ>([float]<SUM>([float]<SQ>([int]$mu))([float]<SUM>([float]<PROD>([int]3)([int]$mu))([int]2))))([float]<SUM>([float]<PROD>([int]3)([int]$mu))([int]2)))')
T1 = GP.tree('([float]<PROD>([int]3)([int]2))')
T2 = GP.tree('([int]5)')
T3 = GP.tree('([float]<PROD>([int]7)([int]11))')
T4 = GP.tree('([int]13)')
T5 = GP.tree('([float]<SUM>([float]<SQ>([float]<SUM>([float]<SQ>([int]17))([float]<SUM>([float]<PROD>([int]3)([int]19))([int]2))))([float]<SUM>([float]<PROD>([int]3)([int]23))([int]2)))')
TF = TestTreeFactory(T0)

GP2 = GPTreebank(tree_factory=DummyTreeFactory(), mutation_rate=0.2, mutation_sd=1.0, crossover_rate=0.5, max_depth=10, max_size=50)
T6  = GP2.tree('([tuple]([int]8)([float]14.0)([complex]35.0+19.0j)([bool]True))')
T7  = GP2.tree('([tuple]([int]-1)([float]1.0)([complex]10.0+1.0j)([bool]False))')
T8  = GP2.tree('([tuple]([int]1)([float]3.0)([complex]20.0+5.0j)([bool]False))')
assert T6() == (8, 14.0, (35+19j), True)


def to_list(self_val, *child_vals, **kwargs):
    self_val = [self_val]
    child_vals = [[val] if isinstance(val, str) else val for val in child_vals]
    return reduce(lambda x, y: x+y, child_vals + [self_val])

def get_credit_pos(tree, pos):
    return tree.metadata['credit'][pos]

tree_lists = [
    [T0, T1, T2],
    [T0, T2],
    [T0, T1],
    [T1, T2],
    [T1, T3],
    [T2, T4],
    [T0],
    [T1],
    [T2]
]


_opf = ops.OperatorFactory()
RTF = RandomTreeFactory([], [], _opf, treebank=GP2)
TS = [
    RTF.NTTemplate(float, _opf('SUM'), [float, float]),
    RTF.NTTemplate(float, _opf('SUM'), [int, int]),
    RTF.NTTemplate(float, _opf('POW'), [float, int]),
    RTF.NTTemplate(float, _opf('TERN_FLOAT'), [bool, float, float]),
    RTF.NTTemplate(int, _opf('INT_SUM'), [int, int]),
    RTF.NTTemplate(bool, _opf('EQ'), [int, int]),
    RTF.NTTemplate(bool, _opf('EQ'), [float, float]),
    RTF.ConstTemplate(float, lambda: 2.0),
    RTF.ConstTemplate(int, lambda: 7),
    RTF.ConstTemplate(bool, lambda: True),
    RTF.ConstTemplate(bool, lambda: False),
    RTF.VarTemplate(float, 'x')
]

