import unittest
from test_materials import paramarama, shhhh
from philoso_py import ModelFactory
import logging
import sys
from collections import defaultdict
from guardrails import Interval

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
                "obs_log_radius"
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
            "log_mutation_sd": Interval("[-11.512925464970229, 0.0]"),
            "max_depth": Interval("[1, inf]"),
            "elitism": Interval("[0, 1]"),
            "temp_coeff": Interval("[0, inf]"),
            "mutator_wt_0": Interval("[0, 1]"),
            "mutator_wt_1": Interval("[0, 1]"),
            "obs_centre": Interval("[-50, 50]"),
            "obs_log_radius": Interval("[-inf, 4.605170185988092]"),
            "sra_tf_log_float_const_sd": Interval("[-11.512925464970229, 0.0]")
        }
        for name, gr in grm.items():
            self.assertEqual(grm[name].raw_interval, Interval("[-10, 10]"))
            self.assertEqual(grm[name].interval, intervals[name])
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
        # self.nums_tf_params = [len(tf.tf_guardrail_params) for tf in self.tf_options]
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

    def test_gp_continue(self):
        gp_continue = self.get_action('gp_continue')
        self.assertTrue(1==1)

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