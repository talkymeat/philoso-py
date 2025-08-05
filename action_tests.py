import unittest
from test_materials import paramarama, shhhh, DummyTreebank
from philoso_py import ModelFactory
from action import scale_unit_to_range
import logging
import sys
from collections import defaultdict
from guardrails import Interval, TanhGuardrail, ExponentialGuardrail
import torch
from collections import OrderedDict
import numpy as np

PARAM_NAMES_NEW = (
    'gp_register', # own head
    'tf_choices', # --
    'tf_weights', # --
    'all_tf_params', # 11 but does nothing
    'pop', # 0-2
    'crossover_rate', # 3
    'mutation_rate', # 4
    'mutation_sd', # 5 but does nothing ;; np.exp(log_mutation_sd), # mutation_sd
    'temp_coeff', # 8
    'max_depth', # 6
    'max_size', # 0-2
    'elitism', # 7
    'episode_len', # 0-2
    'sb_weights', # --
    'obs_args', # 9-10: 9 loc, 10 log rad
    'mut8or_weights' # 12-13
)

PARAM_NAMES_CONTINUE = (
    'gp_register', 
    'crossover_rate',
    'mutation_rate',
    'mutation_sd',
    'temp_coeff',
    'elitism',
    'mut8or_weights'
)

class TestActions(unittest.TestCase):
    def __init__(self, methodName = "runTest"):
        super().__init__(methodName)
        self.mf = ModelFactory()
        self.log = logging.getLogger( "TestActions" )

    @shhhh
    def make_model_from_json(self, params=None):
        params = paramarama if params is None else params
        return self.mf.from_json(params)

    def get_action(self, act_name):
        m = self.make_model_from_json()
        return m.agents[0].actions[act_name]

    def test_gp_new_gp_lists_init(self):
        gp_new = self.get_action('gp_new')
        self.assertEqual(len(gp_new.gptb_list), 2)
        for gptb in gp_new.gptb_list:
            self.assertIsNone(gptb)
        self.assertEqual(len(gp_new.gptb_cts), 2)
        for gptct in gp_new.gptb_cts:
            self.assertEqual(gptct, 0)

    def test_gp_new_gp_vars_out(self):
        gp_new = self.get_action('gp_new')
        self.assertEqual(
            set(gp_new.gp_vars_out),
            {
                "mse",
                "rmse",
                "size",
                "depth",
                "raw_fitness",
                "fitness",
                "value",
                "r",
                "crossover_rate",
                "mutation_rate",
                "mutation_sd",
                "max_depth",
                "max_size",
                "temp_coeff",
                "bounded_sra_tf_float_const_sd",
                "pop",
                "elitism",
                "obs_centre",
                "obs_radius"
            },
        )

    def test_gp_new_guardrail_setup(self):
        gp_new = self.get_action('gp_new')
        grm = gp_new.ac.guardrail_manager
        intervals = {
            "pop": Interval("[2, inf]"),
            "max_size": Interval("[3, inf]"),
            "episode_len": Interval("[2, inf]"),
            "crossover_rate": Interval("[0, 1]"),
            "mutation_rate": Interval("[0, 1]"),
            "mutation_sd": Interval("[1e-05, 1.0]"),
            "max_depth": Interval("[1, inf]"),
            "elitism": Interval("[0, 1]"),
            "temp_coeff": Interval("[0, inf]"),
            "mutator_wt_0": Interval("[0, 1]"),
            "mutator_wt_1": Interval("[0, 1]"),
            "obs_centre": Interval("[-50, 50]"),
            "obs_radius": Interval("[0.01, 100]"),
            "sra_tf_const_sd": Interval("[1e-05, 1.0]")
        }
        exp_vars = ['mutation_sd', 'obs_radius', 'sra_tf_const_sd']
        for name, gr in grm.items():
            if name in exp_vars:
                self.assertIsInstance(gr, ExponentialGuardrail, name)
                self.assertEqual(gr.raw_interval, Interval("[-103.97209, 88.72284]"), name)
            else:
                self.assertIsInstance(gr, TanhGuardrail, name)
                self.assertEqual(gr.raw_interval, Interval("[-10, 10]"), name)
            self.assertEqual(gr.interval, intervals[name], name)
        # self.sb_factory: SimpleGPScoreboardFactory = self.ac.sb_factory
        # # note that the defined fitness value always has a raw weight of 1
        # self.num_sb_weights: int = self.sb_factory.num_sb_weights - 1
        # if CompositeTreeFactory not in tree_factory_classes:
        #     self.tf_options = tree_factory_classes
        # else:
        #     raise ValueError("CompositeTreeFactory cannot be included in tree_factory options")
        # self.num_tf_opts = (
        #     len(self.tf_options) if len(self.tf_options) > 1 else 0
        # ) 
        # self.nums_tf_params = [len(tf.param_specs) for tf in self.tf_options]
        # self.world = world
        # self.num_wobf_params = len(self.world.wobf_param_ranges)
        # self.rng = rng
        # self.out_dir = out_dir if isinstance(out_dir, Path) else Path(out_dir)
        # self.t = time
        # self.dv = dv 
        # self.log_range_mutation_sd = (
        #     np.log(range_mutation_sd[0]),
        #     np.log(range_mutation_sd[1])
        # )
        # self.mutators = mutators 
        # self.num_mut_wts = (
        #     len(self.mutators) if len(self.mutators) > 1 else 0
        # )
        # self.def_fitness = def_fitness
        # # XXX TODO make these size factors user-settable params (not less than 2,3,2 tho)
        # self.min_pop = 2
        # self.min_max_size = 3
        # self.min_ep_len = 2
        # self.size_factor_cuboid = Cuboid(
        #     self.min_pop,
        #     self.min_max_size,
        #     self.min_ep_len,
        #     max_volume
        # )
        # self.theta = theta
        # self.ping_freq = ping_freq
        # self.set_guardrails()

    def test_gp_new_process_action(self):
        gp_new = self.get_action('gp_new')
        action = OrderedDict({
            'gp_register': torch.tensor([0], dtype=torch.int32), 
            'tanh_box': torch.zeros((1, 11), dtype=torch.float32), 
            'exp_box': torch.zeros((1, 3), dtype=torch.float32)
        })
        i = 10 # 11 # 10 # 5
        action['tanh_box'][0][8] = .5
        action['exp_box'][0][0] = np.log(0.1)
        action['exp_box'][0][1] = 1.0
        action['exp_box'][0][2] = np.log(0.01)
        act_params = gp_new.process_action(action, log=self.log.debug)
        params = dict(zip(PARAM_NAMES_NEW, act_params))
        self.assertEqual(params['gp_register'], 0)
        self.assertIsNone(params['tf_choices'])
        self.assertIsNone(params['tf_weights'])
        self.assertEqual(params['pop'], params['episode_len'])
        self.assertEqual(params['pop']+1, params['max_size'])
        self.assertLess(params['pop'] * params['max_size'] * params['episode_len'], gp_new.ac.max_volume)
        self.assertGreater((params['pop']+1) * (params['max_size']+1) * (params['episode_len']+1), gp_new.ac.max_volume)
        self.assertEqual(params['crossover_rate'], 0.5)
        self.assertEqual(params['mutation_rate'], 0.5)
        self.assertEqual(params['temp_coeff'], 1)
        self.assertEqual(params['max_depth'], np.ceil((np.floor(np.log2(params['max_size'])) + params['max_size']/2)/2))
        self.assertEqual(params['elitism'], int(params['pop']/2))
        self.assertEqual(params['sb_weights'], [1])
        self.assertListEqual(list(params['mut8or_weights']), [0.5, 0.5])
        self.assertAlmostEqual(params['all_tf_params'][0]['sra_tf_const_sd'], 0.01)
        self.assertAlmostEqual(params['mutation_sd'], 0.1)
        self.assertAlmostEqual(params['obs_args'][0], np.tanh(.5)*50, places=5)
        self.assertEqual(params['obs_args'][1], np.e)

    def test_gp_new_exp_box(self):
        for _ in range(100):
            randvals = torch.rand((1, 3), dtype=torch.float32)
            gp_new = self.get_action('gp_new')
            action = OrderedDict({
                'gp_register': torch.tensor([0], dtype=torch.int32), 
                'tanh_box': torch.zeros((1, 11), dtype=torch.float32), 
                'exp_box': torch.log(randvals)
            })
            act_params = gp_new.process_action(action, log=self.log.debug)
            params = dict(zip(PARAM_NAMES_NEW, act_params))
            self.assertAlmostEqual(params['mutation_sd'], randvals[0][0]) # 0
            self.assertEqual(params['obs_args'][1], randvals[0][1]) # 1
            self.assertAlmostEqual(params['all_tf_params'][0]['sra_tf_const_sd'], randvals[0][2]) # 2

    def test_gp_continue_process_action(self):
        gp_continue = self.get_action('gp_continue')
        gp_continue.gptb_list[0] = DummyTreebank()
        gp_continue.gptb_cts[0] = 1
        action = OrderedDict({
            'gp_register': torch.tensor([0], dtype=torch.int32), 
            'tanh_box': torch.zeros((1, 6), dtype=torch.float32), 
            'exp_box': torch.zeros((1, 1), dtype=torch.float32)
        })
        # action['tanh_box'][0][8] = .5
        action['exp_box'][0][0] = np.log(0.1)
        act_params = gp_continue.process_action(action, log=self.log.debug)
        params = dict(zip(PARAM_NAMES_CONTINUE, act_params))
        self.assertEqual(params['gp_register'], 0) #
        self.assertEqual(params['crossover_rate'], 0.5) #
        self.assertEqual(params['mutation_rate'], 0.5) #
        self.assertEqual(params['temp_coeff'], 1) #
        self.assertEqual(params['elitism'], gp_continue.gptb_list[0].pop//2) #
        self.assertListEqual(list(params['mut8or_weights']), [0.5, 0.5]) #
        self.assertAlmostEqual(params['mutation_sd'], 0.1) #

    def test_gp_continue_exp_box(self):
        gp_continue = self.get_action('gp_continue')
        for _ in range(100):
            randval = torch.rand((1, 1), dtype=torch.float32)
            gp_continue.gptb_list[0] = DummyTreebank()
            gp_continue.gptb_cts[0] = 1
            action = OrderedDict({
                'gp_register': torch.tensor([0], dtype=torch.int32), 
                'tanh_box': torch.zeros((1, 6), dtype=torch.float32), 
                'exp_box': torch.log(randval)
            })
            act_params = gp_continue.process_action(action, log=self.log.debug)
            params = dict(zip(PARAM_NAMES_CONTINUE, act_params))
            self.assertAlmostEqual(params['mutation_sd'], randval.numpy()[0, 0]) 

    def test_use_mem(self):
        use_mem = self.get_action('use_mem')
        self.assertTrue(1==1)

    def test_store_mem(self):
        store_mem = self.get_action('store_mem')
        self.assertTrue(1==1)

    def test_publish(self):
        publish = self.get_action('publish')
        self.assertTrue(1==1)

    def test_read(self):
        read = self.get_action('read')
        self.assertTrue(1==1)


if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "TestActions" ).setLevel( logging.DEBUG )
    unittest.main()