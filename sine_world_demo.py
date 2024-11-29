import numpy as np
from pathlib import Path

from philoso_py import Model
from world import SineWorld
from observatories import SineWorldObservatoryFactory
from gp_fitness import SimplerGPScoreboardFactory
from model_time import ModelTime
from repository import Publication
from agent import Agent
from agent_controller import AgentController
from tree_factories import SimpleRandomAlgebraicTreeFactory
from mutators import single_leaf_mutator_factory, single_xo_factory
from ppo import ActorCriticNetworkTanh
from reward import Curiosity, Renoun, GuardrailCollisions

dancing_chaos_at_the_heart_of_the_world = np.random.Generator(np.random.PCG64())
print(f'Seed: {dancing_chaos_at_the_heart_of_the_world.bit_generator.seed_seq.entropy}')
out_dir = Path('output', 'test')
ping_freq = 10


world = SineWorld(
    np.pi*5, 100, 0.05, (1,100), (0.1, 10), 
    seed=dancing_chaos_at_the_heart_of_the_world
)
obs_factory = SineWorldObservatoryFactory(world)
sb_factory = SimplerGPScoreboardFactory(
    ['irmse', 'size', 'depth', 'penalty', 'hasnans', 'fitness'],
    'y'
)
n_agents = 8
gp_vars_core = [
    'mse', 'rmse', 'size', 'depth', 'raw_fitness', 'fitness', 'value'
]
gp_vars_more = [
    "crossover_rate", "mutation_rate", 
    "mutation_sd", "max_depth", "max_size", "temp_coeff", "pop", "elitism", 
    'obs_start', 'obs_stop', 'obs_num'
]
time = ModelTime()
agent_names = {f'ag{i}': i for i in range(n_agents)}
pub = Publication(
    gp_vars_core + gp_vars_more, # cols: Sequence[str],
    10, # rows: int,
    time, # model_time: ModelTime,
    agent_names,
    types = np.float64, # types: Sequence[dtype] | Mapping[str, dtype] | dtype | None = None,
    tables = 2, # tables: int = 1,
    reward = 'ranked', # reward: PublicationRewardFunc | str | None = None,
    value = 'value'
    # DEFAULTS USED decay: float = 0.95, value: str = "fitness",
)
agents = [
    Agent(
        AgentController(
            world, # World,
            time, # ModelTime,
            name, # name,
            6, # mem_rows,
            3, # mem_tables,
            world.dv, # dv,
            'irmse', # str, def_fitness
            sb_factory, # SimpleGPScoreboardFactory, # Needs to be more general XXX TODO
            obs_factory, # ObservatoryFactory
            [SimpleRandomAlgebraicTreeFactory], #tree_factory_classes, # tree_factory_classes: list[type[TreeFactory]],
            dancing_chaos_at_the_heart_of_the_world, # np.random.Generator,
            agent_names, #agent_names, # dict[str, int],
            pub, #repository, # Publication,
            out_dir, # out_dir: str|Path,
            50, #record_obs_len, # int,
            max_readings=3, # max_readings, # int = 5,
            mem_col_types=np.float64, # Sequence[np.dtype]|Mapping[str, np.dtype]|np.dtype|None=None,
            gp_vars_core=gp_vars_core,
            gp_vars_more=gp_vars_more,
            ping_freq=ping_freq,
            value='value',
            mutators=[single_leaf_mutator_factory, single_xo_factory]
        ), # AgentController
        dancing_chaos_at_the_heart_of_the_world, # rng
        network_class = ActorCriticNetworkTanh
    ) for name in agent_names.keys()
]
# Note, this must be done after all agents have been made,
# as some params depend on knowing how many other agents there are
for agent in agents:
    agent.make_networks()
model = Model(
    world, #: World,
    dancing_chaos_at_the_heart_of_the_world, #: np.random.Generator,
    agents, #: Container[Agent],
    pub, # publications, #: Publication=None,
    sb_factory, #: SimpleGPScoreboardFactory=None,
    time, #: ModelTime=None
)
for agent in agents:
    agent.ac.model = model
model.add_reward(
    Curiosity(
        model, 'fitness', 1.0
    )
)
model.add_reward(
    Renoun(
        model
    )
)
model.add_reward(
    GuardrailCollisions(
        model
    )
)

model.run(50,100, prefix='j__')



