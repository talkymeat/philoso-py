from gp import GPTreebank
import operators as ops
from functools import reduce
from icecream import ic
from tree_factories import TestTreeFactory
from gp_poly_test import EXAMPLE_POLY_OBS_FAC as POF
from gp_fitness import SimpleGPScoreboardFactory
from repository import Archive, Publication
import numpy as np
from model_time import ModelTime



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

    def set_treebank(self, treebank, *args, **kwargs):
        self.treebank = treebank

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



