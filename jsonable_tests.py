import unittest
import sys
import io

from jsonable import *
from agent_controller import AgentController
from agent import Agent
from gp_fitness import SimplerGPScoreboardFactory
from philoso_py import ModelFactory
from repository import Publication
from reward import Curiosity, Renoun, GuardrailCollisions
from world import SineWorld
from model_time import ModelTime
from hd import HierarchicalDict as HD
import numpy as np
from gp import GPTreebank


def shhhh(test):
    """Decorator to make sure tests run without the classes
    being tested outputting printout
    """
    def shushed(*args, **kwargs):
        # Suppress unwanted printout:
        suppress_text = io.StringIO()
        sys.stdout = suppress_text
        sys.stderr = suppress_text
        # run test
        test(*args, **kwargs)
        # Release output streams:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    return shushed

paramarama = {
    "seed": 666,
    "iv": "x",
    "dv": "y",
    "dtype": "float32",
    "def_fitness": "irmse",
    "out_dir": "output/l/",
    "ping_freq": 5,
    "output_prefix": "l__",
    "days": 100, 
    "steps_per_day": 100,
    "world": "SineWorld",
    "world_params": {
        "radius": 50,
        "max_observation_size": 100,
        "noise_sd": 0.05,
        "sine_wave_params": [
            [
                10,
                100,
                0
            ],
            [
                0.1, 
                1, 
                0
            ]
        ]
    },
    "sb_factory": "SimplerGPScoreboardFactory",
    "sb_factory_params": {
        "best_outvals": [
            "irmse", 
            "size", 
            "depth", 
            "penalty", 
            "hasnans", 
            "fitness"
        ]
    },
    "gp_vars_core": [
        "mse", 
        "rmse", 
        "size", 
        "depth", 
        "raw_fitness", 
        "fitness", 
        "value"
    ],
    "gp_vars_more": [
        "crossover_rate", 
        "mutation_rate", 
        "mutation_sd", 
        "max_depth", 
        "max_size", 
        "temp_coeff", 
        "pop", 
        "elitism", 
        "obs_start", 
        "obs_stop", 
        "obs_num"
    ],
    "publication_params": {
        "rows": 10,
        "tables": 2,
        "reward": "ranked",
        "value": "value",
        # "types": "float 64"
    },
    "agent_populations": [
        "a"
    ],
    "agent_templates": {
        "a": {
            "n": 8,
            "network_class": "ActorCriticNetworkTanh",
            "device": "cpu",
            "controller": {
                "mem_rows": 6,
                "mem_tables": 3,
                "tree_factory_classes": [
                    "SimpleRandomAlgebraicTreeFactory"
                ],
                "record_obs_len": 50,
                "max_readings": 3,
                # "mem_col_types": "float 64",
                "value": "value",
                "mutators": [
                    {
                        "name": "single_leaf_mutator_factory"
                    }, 
                    {
                        "name": "single_xo_factory"
                    }
                ],
                "num_treebanks": 2,
                "short_term_mem_size": 5,
                "max_volume": 50000,
                "max_max_size": 400, 
                "max_max_depth": 100,
                "theta": 0.05,
                "gp_system": "GPTreebank",
                "sb_statfuncs": [
                    {
                        "name": "mean"
                    },
                    {
                        "name": "mode"
                    },
                    {
                        "name": "std"
                    },
                    {
                        "name": "nanage"
                    },
                    {
                        "name": "infage"
                    }
                ],
                "sb_statfuncs_quantiles": 9
            },
            "network_params": {
                "ppo_clip_val": 0.2,
                "target_kl_div": 0.01,
                "max_policy_train_iters": 80,
                "value_train_iters": 80,
                "policy_lr": 3e-4,
                "value_lr": 1e-2
            }
        }
    },
    "rewards": [
        "Curiosity",
        "Renoun",
        "GuardrailCollisions"
    ],
    "reward_params": {
        "Curiosity": {
            "def_fitness": "fitness",
            "first_finding_bonus": 1.0
        },
        "Renoun": {},
        "GuardrailCollisions": {}
    }
}

class TestJSONables(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.mf = ModelFactory()
        self.paramarama = self.mf._read_json(paramarama)

    def test_read_json(self):
        self.assertIsInstance(self.mf._read_json(paramarama), HD)

    def make_sb_factory_from_json(self):
        return SimplerGPScoreboardFactory.from_json(self.paramarama)

    def test_sb_factory_from_json(self):
        gpf = self.make_sb_factory_from_json()
        self.assertEqual(gpf.def_fitness, 'irmse')
        self.assertEqual(len(gpf.best_outvals), 6)
        for b_o in ['irmse', 'size', 'depth', 'penalty', 'hasnans', 'fitness']:
            self.assertIn(b_o, gpf.best_outvals)
        self.assertEqual(gpf.dv, 'y')

    def test_sb_factory_json(self):
        gpf = self.make_sb_factory_from_json()
        self.assertEqual(
            gpf.json,
            {
                'def_fitness': 'irmse',
                'best_outvals': [
                    'irmse', 
                    'size', 
                    'depth', 
                    'penalty', 
                    'hasnans', 
                    'fitness'
                ],
                'dv': 'y'
            }
        )

    def make_world_from_json(self):
        return self.mf.worlds[paramarama['world']].from_json(self.paramarama)

    def test_world_from_json(self):
        world = self.make_world_from_json()
        self.assertIsInstance(world, SineWorld)
        self.assertEqual(world.range[1], 50)
        self.assertEqual(world.max_observation_size, 100)
        self.assertAlmostEqual(world.noise_sd, 0.05)
        self.assertEqual(len(world.sine_waves), 2)
        self.assertEqual(world.sine_waves[0].wavelength, 10)
        self.assertEqual(world.sine_waves[0].amplitude, 100)
        self.assertEqual(world.sine_waves[0].phase, 0)
        self.assertAlmostEqual(world.sine_waves[1].wavelength, 0.1)
        self.assertEqual(world.sine_waves[1].amplitude, 1)
        self.assertEqual(world.sine_waves[1].phase, 0)
        self.assertEqual(world.np_random.bit_generator.seed_seq.entropy, 666)
        self.assertEqual(int(world.speed), 0)
        self.assertEqual(str(world.dtype), 'float32')
        self.assertEqual(world.iv, 'x')
        self.assertEqual(world.dv, 'y')

    def test_gpf_json(self):
        world = self.make_world_from_json()
        self.assertEqual(
            world.json,
            {
                'radius': 50,
                'max_observation_size': 100,
                'noise_sd': 0.05,
                'sine_wave_params': [
                    [10, 100, 0], [0.1, 1, 0]
                    ],
                'seed': 666,
                'speed': 0.0,
                'dtype': 'float32',
                'iv': 'x',
                'dv': 'y'
            }
        )

    def make_network_class(self):
        return self.mf.network_classes[
            self.paramarama[[
                "agent_templates", 
                'a', 
                "controller", 
                "network_class"
            ]]
        ]

    def test_network_class(self):
        network_class = self.make_network_class()
        self.assertEqual(str(network_class.__name__), 'ActorCriticNetworkTanh')

    def make_agent_names(self):
        return {f'a_{i}': i for i in range(8)}

    def make_publication_from_json(self):
        mt = ModelTime()
        agnames = self.make_agent_names()
        return Publication.from_json(self.paramarama, time=mt, agent_names=agnames)

    def test_publication_from_json(self):
        pub = self.make_publication_from_json()
        noncore_df = pub.tables[0].drop(pub.ESSENTIAL_COLS, axis=1)
        self.assertEqual(
            set(noncore_df.columns),
            {
                'mse', 'rmse', 'size', 'depth', 'raw_fitness',
                'fitness', 'value', 'crossover_rate',
                'mutation_rate', 'mutation_sd', 'max_depth',
                'max_size', 'temp_coeff', 'pop', 'elitism',
                'obs_start', 'obs_stop', 'obs_num'
            }
        )
        self.assertEqual(len(noncore_df), 10)
        self.assertEqual(pub.noncore_types, 'float32')
        self.assertEqual(len(pub.tables), 2)
        self.assertEqual(pub.value, 'value')
        self.assertEqual(
            pub.agent_names,
            {
                'a_0': 0, 'a_1': 1, 'a_2': 2, 'a_3': 3,
                'a_4': 4, 'a_5': 5, 'a_6': 6, 'a_7': 7
            }
        )
        self.assertEqual(pub.reward_name, 'ranked')
        self.assertAlmostEqual(pub.decay, 0.95)

    def test_publication_json(self):
        json_ = self.make_publication_from_json().json
        self.assertEqual(
            set(json_['cols']),
            {
                'mse', 'rmse', 'size', 'depth', 'raw_fitness',
                'fitness', 'value', 'crossover_rate',
                'mutation_rate', 'mutation_sd', 'max_depth',
                'max_size', 'temp_coeff', 'pop', 'elitism',
                'obs_start', 'obs_stop', 'obs_num'
            }
        )
        self.assertEqual(json_['rows'], 10)
        # self.assertEqual(json_['types'], 'float 64')
        self.assertEqual(json_['tables'], 2)
        self.assertEqual(json_['value'], 'value')
        self.assertEqual(
            json_['agent_indices'],
            {
                'a_0': 0, 'a_1': 1, 'a_2': 2, 'a_3': 3,
                'a_4': 4, 'a_5': 5, 'a_6': 6, 'a_7': 7
            }
        )
        self.assertEqual(json_['reward'], 'ranked')
        self.assertAlmostEqual(json_['decay'], 0.95)
        self.assertEqual(
            set(json_.keys()), 
            {
                'cols', 'rows', 'tables', 'types', 'value', 
                'agent_indices', 'reward', 'decay'
            }
        )

    def make_mutators(self):
        return [
            self.mf.mutators[mut8['name']]
            for mut8 
            in self.paramarama[[
                "agent_templates", 
                'a', 
                "controller", 
                "mutators"
            ]]
        ]

    def test_mutators(self):
        mutators = self.make_mutators()
        self.assertEqual(
            [str(mut8) for mut8 in mutators],
            [
                'single_leaf_mutator_factory(*args, **kwargs)', 
                'single_xo_factory(*args, **kwargs)'
            ]
        )

    def make_statfuncs(self):
        sb_statfuncs=[
            self.mf.sb_statfuncs[sbsf['name']]
            for sbsf 
            in self.paramarama.get([
                "agent_templates", 
                'a', 
                "controller", 
                "sb_statfuncs"
            ], [])
        ]
        sb_statfuncs+=[
            self.mf.sb_statfuncs[sbsf['name']](**sbsf)
            for sbsf 
            in [{
                "name": "Quantile",
                "n": 9,
                "i": i
            } for i in range(9)]
        ] 
        return sb_statfuncs   

    def test_statfuncs_from_json(self):
        sb_statfuncs=self.make_statfuncs()
        self.assertEqual(
            [repr(sbf) for sbf in sb_statfuncs],
            [
                'mean(*args, **kwargs)',
                'mode(*args, **kwargs)',
                'std(*args, **kwargs)',
                'nanage(*args, **kwargs)',
                'infage(*args, **kwargs)',
                'Quantile(n=9, i=0)',
                'Quantile(n=9, i=1)',
                'Quantile(n=9, i=2)',
                'Quantile(n=9, i=3)',
                'Quantile(n=9, i=4)',
                'Quantile(n=9, i=5)',
                'Quantile(n=9, i=6)',
                'Quantile(n=9, i=7)',
                'Quantile(n=9, i=8)'
            ]
        )

    def test_statfuncs_json(self):
        self.assertEqual(
            [sbsf.json for sbsf in self.make_statfuncs()],
            [
                {'name': 'mean'},
                {'name': 'mode'},
                {'name': 'std'},
                {'name': 'nanage'},
                {'name': 'infage'},
                {'name': 'Quantile', 'n': 9, 'i': 0},
                {'name': 'Quantile', 'n': 9, 'i': 1},
                {'name': 'Quantile', 'n': 9, 'i': 2},
                {'name': 'Quantile', 'n': 9, 'i': 3},
                {'name': 'Quantile', 'n': 9, 'i': 4},
                {'name': 'Quantile', 'n': 9, 'i': 5},
                {'name': 'Quantile', 'n': 9, 'i': 6},
                {'name': 'Quantile', 'n': 9, 'i': 7},
                {'name': 'Quantile', 'n': 9, 'i': 8}
            ]
        )

    def make_tree_factory_classes(self):
        return [
            self.mf.tree_factory_classes[tfc]
            for tfc 
            in self.paramarama[[
                "agent_templates", 
                'a', 
                "controller", 
                "tree_factory_classes"
            ]]
        ]

    def test_tree_factory_classes(self):
        tree_factory_classes = self.make_tree_factory_classes()
        self.assertEqual(
            tree_factory_classes[0].__name__,
            'SimpleRandomAlgebraicTreeFactory'
        )

    def make_gp_class(self):
        return self.mf.gp_systems[
            self.paramarama[[
                "agent_templates", 
                'a', 
                "controller", 
                "gp_system"
            ]]
        ]

    def test_gp_class(self):
        gp_system = self.make_gp_class()
        self.assertEqual(
            gp_system.__name__,
            'GPTreebank'
        )

    def make_agent_controller_from_json(self):
        mt = ModelTime()
        return AgentController.from_json(
            self.paramarama,
            world=self.make_world_from_json(),
            time=mt,
            name='a_0',
            prefix='a',
            sb_factory=self.make_sb_factory_from_json(),
            tree_factory_classes=self.make_tree_factory_classes(), 
            agent_indices=self.make_agent_names(),
            repository=self.make_publication_from_json(),
            gp_system=self.make_gp_class(),
            mutators=self.mf.mutators,
            sb_statfuncs=self.mf.sb_statfuncs,
            rng = np.random.Generator(np.random.PCG64(666))
        )

    def test_agent_controller_from_json(self):
        ac = self.make_agent_controller_from_json()
        self.assertIsInstance(ac.world, SineWorld)
        self.assertIsNone(ac.model)
        self.assertEqual(ac._np_random.bit_generator.seed_seq.entropy, 666)
        self.assertEqual(ac.name, 'a_0')
        self.assertEqual(ac.prefix, 'a')
        self.assertEqual(ac.dv, 'y')
        self.assertEqual(str(ac.out_dir), 'output/l/a_0')
        self.assertIs(ac.gp_system, GPTreebank)
        self.assertEqual(ac.def_fitness, 'irmse')
        self.assertEqual(ac.max_volume, 50000)
        for gptb in ac.gptb_list:
            self.assertIsNone(gptb)
        self.assertIsInstance(ac.sb_factory, SimplerGPScoreboardFactory)
        self.assertDictEqual(ac.agent_names, self.make_agent_names())
        self.assertAlmostEqual(ac.theta, 0.05)
        self.assertEqual(ac.ping_freq, 5)
        self.assertEqual(ac.short_term_mem_size, 5)
        self.assertEqual(ac.record_obs_len, 50)
        self.assertEqual(
            ac.guardrail_manager['_'].__class__.__name__, 
            "NoGuardrail"
        )
        self.assertCountEqual(
            [repr(mut) for mut in ac.mutators],
            [
                'single_leaf_mutator_factory(*args, **kwargs)',
                'single_xo_factory(*args, **kwargs)'
            ]
        )
        self.assertCountEqual(
            ac.gp_vars_core,
            [
                'mse',
                'rmse',
                'size',
                'depth',
                'raw_fitness',
                'fitness',
                'value'
            ]
        )
        self.assertCountEqual(
            ac.gp_vars_more,
            [
                'crossover_rate',
                'mutation_rate',
                'mutation_sd',
                'max_depth',
                'max_size',
                'temp_coeff',
                'pop',
                'elitism',
                'obs_start',
                'obs_stop',
                'obs_num'
            ]
        )
        self.assertCountEqual(
            [repr(sbsf) for sbsf in ac.sb_statfuncs],
            [
                'mean(*args, **kwargs)',
                'mode(*args, **kwargs)',
                'std(*args, **kwargs)',
                'nanage(*args, **kwargs)',
                'infage(*args, **kwargs)',
                'Quantile(n=9, i=0)',
                'Quantile(n=9, i=1)',
                'Quantile(n=9, i=2)',
                'Quantile(n=9, i=3)',
                'Quantile(n=9, i=4)',
                'Quantile(n=9, i=5)',
                'Quantile(n=9, i=6)',
                'Quantile(n=9, i=7)',
                'Quantile(n=9, i=8)'
            ]
        )
        self.assertCountEqual(
            ac.memory.cols, ac.gp_vars_core+ac.gp_vars_more
        )
        self.assertEqual(ac.memory.rows, 6)
        self.assertEqual(len(ac.memory.tables), 3)
        self.assertEqual(ac.memory.value, 'value')
        self.assertEqual(ac.memory.max_size, 400)
        self.assertEqual(ac.memory.max_depth, 100)
        self.assertEqual(ac.max_readings, 3)
        self.assertCountEqual(
            [tfc.__name__ for tfc in ac.tree_factory_classes],
            ['SimpleRandomAlgebraicTreeFactory']
        )
        self.assertFalse(ac.args)
        self.assertFalse(ac.kwargs)

    def make_model_from_json(self, params=None):
        params = self.paramarama if params is None else params
        return self.mf.from_json(params)

    @shhhh
    def test_model_from_json(self):
        m1 = self.make_model_from_json()
        self.assertIsInstance(m1.world, SineWorld)
        self.assertEqual(len(m1.agents), 8)
        for agent in m1.agents:
            self.assertIsInstance(agent, Agent)
        self.assertEqual(
            m1.agent_name_set,
            {'a_0', 'a_1', 'a_2', 'a_3', 'a_4', 'a_5', 'a_6', 'a_7'}
        )
        self.assertIsInstance(m1.publications, Publication)
        self.assertEqual(
            self.paramarama['seed'],
            m1.rng.bit_generator.seed_seq.entropy
        )
        self.assertEqual(len(m1.rewards), 3)
        rset = set()
        for reward in m1.rewards:
            if isinstance(reward, Curiosity):
                rset.add(0)
                self.assertEqual(reward.def_fitness, "fitness")
                self.assertEqual(reward.first_finding_bonus, 1.0)
            elif isinstance(reward, Renoun):
                rset.add(1)
            elif isinstance(reward, GuardrailCollisions):
                rset.add(2)
        self.assertEqual(len(rset), 3)
        self.assertEqual(m1.out_dir, 'output/l/')

    @shhhh
    def test_model_json(self):
        mjson = HD(self.make_model_from_json().json)
        for j in range(3):
            self.assertEqual(mjson['out_dir'], 'output/l/')
            self.assertEqual(mjson['world'], 'SineWorld')
            self.assertEqual(mjson[['world_params', 'radius']], 50)
            self.assertEqual(mjson[['world_params', 'max_observation_size']], 100)
            self.assertAlmostEqual(mjson[['world_params', 'noise_sd']], 0.05)
            self.assertEqual(mjson[['world_params', 'sine_wave_params']][0], [10, 100, 0])
            self.assertAlmostEqual(mjson[['world_params', 'sine_wave_params']][1][0], 0.1)
            self.assertEqual(mjson[['world_params', 'sine_wave_params']][1][1:], [1, 0])
            self.assertAlmostEqual(mjson[['world_params', 'speed']], 0.0)
            self.assertEqual(mjson[['world_params', 'dtype']], 'float32')
            self.assertEqual(mjson[['world_params', 'iv']], 'x')
            self.assertEqual(mjson['sb_factory'], 'SimplerGPScoreboardFactory')
            self.assertCountEqual(
                mjson[['sb_factory_params', 'best_outvals']], 
                ['irmse', 'size', 'depth', 'penalty', 'hasnans', 'fitness']
            )
            self.assertCountEqual(
                mjson[['publication_params', 'cols']], 
                [
                    'mse', 'rmse', 'size', 'depth', 'raw_fitness', 'fitness', 
                    'value', 'crossover_rate', 'mutation_rate', 'mutation_sd', 
                    'max_depth', 'max_size', 'temp_coeff', 'pop', 'elitism', 
                    'obs_start', 'obs_stop', 'obs_num'
                ]
            )
            self.assertEqual(mjson[['publication_params', 'rows']], 10)
            # self.assertEqual(mjson[['publication_params', 'types']], 'float 64')
            self.assertEqual(mjson[['publication_params', 'tables']], 2)
            self.assertEqual(mjson[['publication_params', 'reward']], 'ranked')
            self.assertAlmostEqual(mjson[['publication_params', 'decay']], 0.95)
            self.assertEqual(mjson['agent_populations'], ['a'])
            self.assertEqual(mjson['agent_names'], [])
            self.assertIn(['agent_templates', 'a'], mjson)
            self.assertIn(['agent_templates', 'a', 'controller'], mjson)
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'mem_rows']], 6
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'mem_tables']], 3
            )
            self.assertCountEqual(
                mjson[['agent_templates', 'a', 'controller', 'tree_factory_classes']], 
                ['SimpleRandomAlgebraicTreeFactory']
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'out_dir']], 'output/l'
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'record_obs_len']], 50
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'max_readings']], 3
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'mem_col_types']], 'float32'
            )   
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'gp_system']], 'GPTreebank'
            )   
            self.assertCountEqual(
                mjson[['agent_templates', 'a', 'controller', 'sb_statfuncs']], 
                [
                    {'name': 'mean'}, {'name': 'mode'}, {'name': 'std'}, 
                    {'name': 'nanage'}, {'name': 'infage'}, 
                    {'name': 'Quantile', 'n': 9, 'i': 0}, 
                    {'name': 'Quantile', 'n': 9, 'i': 1}, 
                    {'name': 'Quantile', 'n': 9, 'i': 2}, 
                    {'name': 'Quantile', 'n': 9, 'i': 3}, 
                    {'name': 'Quantile', 'n': 9, 'i': 4}, 
                    {'name': 'Quantile', 'n': 9, 'i': 5}, 
                    {'name': 'Quantile', 'n': 9, 'i': 6}, 
                    {'name': 'Quantile', 'n': 9, 'i': 7}, 
                    {'name': 'Quantile', 'n': 9, 'i': 8}
                ]
            )   
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'num_treebanks']], 2
            )   
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'short_term_mem_size']], 5
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'max_volume']], 50000
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'max_max_size']], 400
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'max_max_depth']], 100
            )
            self.assertAlmostEqual(
                mjson[['agent_templates', 'a', 'controller', 'theta']], 0.05 
            )
            self.assertCountEqual(
                mjson[['agent_templates', 'a', 'controller', 'gp_vars_core']], 
                ['mse', 'rmse', 'size', 'depth', 'raw_fitness', 'fitness', 'value']
            )
            self.assertCountEqual(
                mjson[['agent_templates', 'a', 'controller', 'gp_vars_more']], 
                [
                    'crossover_rate', 'mutation_rate', 'mutation_sd', 'max_depth', 
                    'max_size', 'temp_coeff', 'pop', 'elitism', 'obs_start', 
                    'obs_stop', 'obs_num'
                ]
            )
            self.assertAlmostEqual(
                mjson[['agent_templates', 'a', 'controller', 'guardrail_base_penalty']], 1.0
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'ping_freq']], 5
            )
            self.assertCountEqual(
                mjson[['agent_templates', 'a', 'controller', 'mutators']], 
                [{'name': 'single_leaf_mutator_factory'}, {'name': 'single_xo_factory'}]
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'args']], ()
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'controller', 'kwargs']], {}
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'device']], 'cpu'
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'network_class']], 'ActorCriticNetworkTanh'
            )
            self.assertAlmostEqual(
                mjson[['agent_templates', 'a', 'network_params', 'ppo_clip_val']], 0.2
            )
            self.assertAlmostEqual(
                mjson[['agent_templates', 'a', 'network_params', 'target_kl_div']], 0.01
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'network_params', 'max_policy_train_iters']], 80
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'network_params', 'value_train_iters']], 80
            )
            self.assertAlmostEqual(
                mjson[['agent_templates', 'a', 'network_params', 'policy_lr']], 0.0003
            )
            self.assertAlmostEqual(
                mjson[['agent_templates', 'a', 'network_params', 'value_lr']], 0.01
            )
            self.assertEqual(
                mjson[['agent_templates', 'a', 'n']], 8
            )
            reward_names = ['Curiosity', 'Renoun', 'GuardrailCollisions']
            self.assertCountEqual(
                mjson[['rewards']], reward_names
            )
            for name in reward_names:
                self.assertEqual(mjson[['reward_params', name, 'name']], name)
                if name=='Curiosity':
                    self.assertAlmostEqual(
                        mjson[['reward_params', name, 'first_finding_bonus']], 1.0
                    )
                    self.assertEqual(
                        mjson[['reward_params', name, 'def_fitness']], 'fitness'
                    )
            self.assertEqual(mjson[['seed']], 666)
            self.assertEqual(mjson[['dv']], 'y')
            self.assertEqual(mjson[['def_fitness']], 'irmse')
            self.assertEqual(mjson[['value']], 'value')
            self.assertEqual(len(mjson[['agent_indices']]), 8)
            for i in range(8):
                self.assertEqual(mjson[['agent_indices', f'a_{i}']], i)
            if j < 2:
                mjson_new = HD(self.make_model_from_json(mjson).json)
                self.assertEqual(mjson, mjson_new)
                mjson = mjson_new

if __name__ == '__main__':
    unittest.main()