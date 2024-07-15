import gp
import torch
import numpy as np
from gp_fitness import SimpleGPScoreboardFactory
from observatories import SineWorldObservatoryFactory
from world import SineWorld
from utils import _i as __i
from tree_factories import RandomPolynomialFactory

world = SineWorld(
    np.pi*5, 100, 0.05, (1,1), (0.1, 0.1)
)
obs_factory = SineWorldObservatoryFactory(world)
dancing_chaos = np.random.Generator(np.random.PCG64(42))
sb_factory = SimpleGPScoreboardFactory(
    ['irmse', 'size', 'depth', 'penalty', 'hasnans', 'fitness'], 'y'
)
sb_weights=np.array([0.5, 0. , 0.5])
obs_args=(torch.tensor(-15.7080, dtype=torch.float64), torch.tensor(15.7080, dtype=torch.float64), torch.tensor(100., dtype=torch.float64))
observatory = obs_factory(*[__i(arg) for arg in obs_args])
scoreboard = sb_factory(observatory, 2.0, sb_weights)
tree_factory = RandomPolynomialFactory(seed=dancing_chaos, params=None)
tree_factory2 = RandomPolynomialFactory(seed=dancing_chaos, params=None)
gp1 = gp.GPTreebank(
    pop=8333, max_size=3, episode_len=2, max_depth=1, crossover_rate=1.0, 
    mutation_rate=1.0, mutation_sd=0.0, temp_coeff=2.0, elitism=0, ping_freq=1,
    observatory=observatory, seed=dancing_chaos, fitness=scoreboard, 
    operators=tree_factory.op_set, tree_factory=tree_factory
)
gp2 = gp.GPTreebank(
    pop=8333, max_size=3, episode_len=2, max_depth=1, crossover_rate=1.0, 
    mutation_rate=1.0, mutation_sd=0.0, temp_coeff=2.0, elitism=0, ping_freq=1,
    observatory=observatory, seed=dancing_chaos, fitness=scoreboard, 
    operators=tree_factory.op_set, tree_factory=tree_factory2
)

t0 = gp2.tree("([float]<SUM>([float]0.13399396760214105)([float]-0.518961717560086))")
t1 = gp2.tree("([float]<SUM>([float]-0.1850403336787581)([float]-0.011904015822323089))")
t2 = gp2.tree("([float]<SUM>([float]1.0999280132132936)([float]-0.9108169004844523))")
t3 = gp2.tree("([float]<SUM>([float]0.9341657427811247)([float]-0.8601801380721525))")

gp1.set_up_run()
gp1.insert_trees([t0, t1, t2, t3])
gp1.continue_run()