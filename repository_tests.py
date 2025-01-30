import unittest
import numpy as np
from icecream import ic

from repository import *
from test_materials import AG, AGIDXS, TT, T0
from model_time import ModelTime
from utils import aeq

mt=ModelTime()

class TestRepos(unittest.TestCase):
    mem = Archive(
        cols=['goodness'], 
        rows=10,
        model_time=mt,
        types={'goodness': np.float64},
        tables=2,
        value='goodness',
    )
    plos_one = Publication(
        cols=['goodness'], 
        rows=10,
        model_time=mt,
        agent_names=AGIDXS,
        types={'goodness': np.float64},
        tables=2,
        value='goodness',
        reward='ranked'
    )
    agent_nos = [
        3, 5, 2, 4, 
        3, 8, 1, 
        2, 6, 7
    ]
    goodness = [
        0.5, 0.1, 0.4, 0.6, 
        0.8, 0.7, 1.0, 
        0.9, 1.3, 1.2
    ]
    expected_t = [
        8, 9, 6, 7,
        4, 5, 3,
        0, 2, 1
    ]
    expected_rew = [
        0.0, 1.0, 1.8, 2.0,
        1.0, 0.9, 1.0, 
        0.9, 0.9, 0.0
    ]
    exp_agent_order = [
        'a_6', 'a_7', 'a_1', 'a_2', 
        'a_3', 'a_8', 'a_4', 
        'a_3', 'a_2', 'a_5'
    ]

    def __init__(self, methodName = "runTest"):
        mt.reset()
        super().__init__(methodName)
        for ag in AG:
            self.plos_one._add_user(ag)
        for ag, gd, t in zip(self.agent_nos, self.goodness, TT):
            self.plos_one.insert_tree(t.copy(), AG[ag].name, journal=0, goodness=gd)
            mt.tick()

    def test_publication(self):
        self.assertTrue(aeq(self.plos_one._agents['reward'].copy().reset_index(drop=True), (pd.Series(self.expected_rew))).all(), f"""Expected:
        {self.expected_rew}
        Got:
        {list(self.plos_one._agents['reward'])}
        """ 
        )

        self.assertTrue(aeq(self.plos_one.tables[0]['t'].copy().reset_index(drop=True), (pd.Series(self.expected_t))).all(), f"""Expected:
        {self.expected_t}
        Got:
        {list(self.plos_one.tables[0]['t'])}
        """)

        for name, idx in zip(self.exp_agent_order, self.plos_one.tables[0]['credit']):
            self.assertEqual(AG[idx].name, name)

        self.plos_one.insert_tree(T0.copy(), AG[0].name, journal=0, goodness=0.45)
        mt.tick()

        self.assertAlmostEqual(self.plos_one._agents.loc['a_0', 'reward'], 0.2)
        self.assertAlmostEqual(
            self.plos_one._agents.loc['a_5', 'reward'], 
            0.9-((0.95**9)*0.9), 
            f'Oops, {self.plos_one._agents.loc['a_5', 'reward']} != {0.9-((0.95**9)*0.9)}'
        )


if __name__ == '__main__':
    unittest.main()