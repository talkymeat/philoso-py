from enum import Enum
from typing import Any
from collections import OrderedDict
import numpy as np
import pandas as pd
from world import World
from tree_factories import TreeFactory, CompositeTreeFactory, TreeFactoryFactory
from gp import GPTreebank
import operators as ops
from dataclasses import dataclass
from world import World
from trees import Tree
from gymnasium import Env
from gymnasium.spaces import Dict, Tuple, Discrete, Box, MultiBinary, MultiDiscrete, Space
from gymnasium.spaces.utils import flatten, unflatten, flatten_space
from gymnasium.utils import seeding
from rl_bases import Actionable
from gp_fitness import SimpleScoreboardFactory

class Action(Enum):
    RUN_GP = 1
    # REMEMBER_LAST_RESULT = 2
    # PUBLISH_RESULT = 3
    # PUBLISH_BET = 4
    # TAKE_BET = 5
    # LEARN_PUBLISHED_RESULT = 6
    DO_NOTHING = 0



@dataclass
class Result:
    best_tree: Tree 
    fitness: float
    

class Agent(Env):

    def __init__(self, 
            world: World,
            gp_system=type[GPTreebank],
            fitness_measures = None,
            tree_factory_factories: list[TreeFactoryFactory]=None,
            max_actions: int = 40,
            gp_steps: int = 100,
            max_thinking_nodes: int = 20_000,
            max_knowledge_nodes: int = 1000,
            max_knowledge_trees: int = 50,
            max_max_size: int = 120
            # value_measures: list=None,
            # value_measure_weights: list[float]=None,
        ):
        """What should be in __init__, and what in reset?
        """
        self.world = world
        self.gp_system = gp_system
        self.tree_factory_factories = tree_factory_factories
        self.max_actions = max_actions
        self.gp_steps = gp_steps
        self.last_result = {}
        self.num_actions = 0 # reset
        #self.operators = operators
        self.max_thinking_nodes = max_thinking_nodes
        # self.max_learnt_operators = max_learnt_operators
        self.max_knowledge_nodes = max_knowledge_nodes
        self.max_knowledge_trees = max_knowledge_trees
        self.max_max_size = max_max_size
        # self.tree_copy_protocols = tree_copy_protocols
        fm_bools = {
            'use_imse': False,
            'use_irmse': False, 
            'use_isae': False, 
            'use_size': False, 
            'use_depth': False
        }
        self.fitness_measures = fitness_measures if fitness_measures else ['mse']
        for fm in self.fitness_measures:
            if f'use_{fm}' not in fm_bools:
                raise ValueError(
                    f"{fm} is not a valid fitness measure. Philoso.py's GP system" +
                    " cannot yet accommodate custom fitness measures"
                )
            fm_bools[f'use_{fm}'] = True
        self.scoreboard_factory = SimpleScoreboardFactory(**fm_bools)
        max_max_size = min(self.max_thinking_nodes//20, self.max_max_size)
        max_max_depth = max_max_size//2
        self._action_space = Dict({
            # For now, 0=DoNothing, 1=RunGP: will remove DoNothing, add more real action types later
            'actions': MultiBinary(2),
            # Store last result
            'store_last': Discrete(self.max_knowledge_trees),
            # GP hyperparameters 
            'gp_hyperparams': Dict({
                'ranged_hyperparams': Box(
                    low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), # key below VVV
                    high = np.array([
                            1.0,                 # crossover rate
                            1.0,                 # mutation rate
                            np.inf,              # mutation standard deviation
                            float(max_max_size),  # max tree size
                            float(max_max_depth),  # max tree depth
                            1.0 # elitism (replace with tree copy protocols params)
                        ], dtype = np.float32
                    )                 
                ),
                # control ops used in treefactories, if appropriate. get ops from
                # factories
                # "operators": MultiBinary(len(operators)), #+ max_learnt_operators),
                "fitness_weights": self.scoreboard_factory.action_param_space,
                **({"factory_weights": Box(
                    low=0.0, high=1.0, shape=(len(self.tree_factory_factories), ), dtype=np.float32
                )} if len(self.tree_factory_factories)>1 else {}),
                "factory_params": Tuple([
                    tree_fac.action_param_space for tree_fac in self.tree_factory_factories
                ])
            }),
            'observation_params': self.world.action_param_space
        })
        measures = (
            ['raw_fitness'] 
            if len(self.fitness_measures)==1 
            else [fm for fm in self.fitness_measures if fm not in ['size', 'depth']]
        ) 
        self.obs_measures = ['size', 'depth'] + measures + ['hasnans', 'penalty', 'survive', 'fitness']
        self.knowledge_base = self.gp_system(
            max_depth=self.max_knowledge_nodes,
            max_size=self.max_knowledge_nodes, # XXX Operators
        )
        self.knowledge_table = pd.DataFrame(
            columns=['tree', 'exists']+self.obs_measures+self.world.action_param_names,
            index=range(self.max_knowledge_trees)
        ).fillna(0).astype({
            'tree': 'object', 'exists': bool, 'size': 'int16', 'depth': 'int16',
            **{k: 'float32' for k in (measures + ['hasnans', 'penalty', 'survive', 'fitness'])},
            **{k: 'int16' for k in self.world.action_param_names}
        })
        sh = (self.gp_steps,)
        gp_result_observation_space = Dict({
            'size': Box(low=0, high=max_max_size, shape=sh, dtype=np.int16),
            'depth': Box(low=0, high=max_max_depth, shape=sh, dtype=np.int16),
            **{k: Box(low=0.0, high=np.inf, shape=sh) for k in measures},
            'hasnans': Box(low=0.0, high=1.0, shape=sh),
            'penalty': Box(low=0.0, high=np.inf, shape=sh),
            'survive': Box(low=0.0, high=1.0, shape=sh),
            'fitness': Box(low=0.0, high=np.inf, shape=sh)
        })
        knowledge_observation_space = Dict({
            'exists': MultiBinary(self.max_knowledge_trees),
            **{k: Box(
                low=v.low[:self.max_knowledge_trees], 
                high=v.high[:self.max_knowledge_trees], 
                dtype=v.dtype
            ) for k, v in gp_result_observation_space.items()},
            **{k: Box(
                low=low, 
                high=high, 
                shape=(self.max_knowledge_trees,), 
                dtype=self.world.action_param_space.dtype
            ) for k, low, high in zip(
                self.world.action_param_names,
                self.world.action_param_space.low,
                self.world.action_param_space.high
            )}
        })
        self._observation_space = Dict({
            'gp_run': gp_result_observation_space,
            'knowledge': knowledge_observation_space
        })

    @property
    def observation_space(self):
        return flatten_space(self._observation_space)

    @property
    def action_space(self):
        return flatten_space(self._action_space)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Dict, dict[str, Any]]:  # type: ignore
        """Resets the environment to an initial internal state, returning an initial observation and info.

        This method generates a new starting state often with some randomness to ensure that the agent explores the
        state space and learns a generalised policy about the environment. This randomness can be controlled
        with the ``seed`` parameter otherwise if the environment already has a random number generator and
        :meth:`reset` is called with ``seed=None``, the RNG is not reset.

        Therefore, :meth:`reset` should (in the typical use case) be called with a seed right after initialization and then never again.

        For Custom environments, the first line of :meth:`reset` should be ``super().reset(seed=seed)`` which implements
        the seeding correctly.

        .. versionchanged:: v0.25

            The ``return_info`` parameter was removed and now info is expected to be returned.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`).
                If the environment does not already have a PRNG and ``seed=None`` (the default option) is passed,
                a seed will be chosen from some source of entropy (e.g. timestamp or /dev/urandom).
                However, if the environment already has a PRNG and ``seed=None`` is passed, the PRNG will *not* be reset.
                If you pass an integer, the PRNG will be reset even if it already exists.
                Usually, you want to pass an integer *right after the environment has been initialized and then never again*.
                Please refer to the minimal example above to see this paradigm in action.
            options (optional dict): Additional information to specify how the environment is reset (optional,
                depending on the specific environment)

        Returns:
            observation (ObsType): Observation of the initial state. This will be an element of :attr:`_observation_space`
                (typically a numpy array) and is analogous to the observation returned by :meth:`step`.
            info (dictionary):  This dictionary contains auxiliary information complementing ``observation``. It should be analogous to
                the ``info`` returned by :meth:`step`.
        """
        # Initialize the RNG if the seed is manually passed
        super().reset(seed=seed)
        print(f'Seed: {self.np_random.bit_generator.seed_seq.entropy}')
        self.world.seed = self.np_random
        for tff in self.tree_factory_factories:
            tff.seed = self.np_random
        
        # It is hardwired here that the default reward measure is imse, 
        # unless it is not calculated, then irmse, failing that isae
        for fm in ['imse', 'irmse', 'isae']:
            if fm in self.fitness_measures:
                self.gp_reward_measure = fm
                break
        else:
            raise ValueError(
                f"Invalid value for fitness_measures {self.fitness_measures}: " +
                "at least one of 'imse', 'irmse', and 'isae' must be included" 
            )
        return flatten(self._observation_space, self._empty_obs()), {}
    
    def _empty_obs(self):
        observation = OrderedDict({
            k0: {
                k1: np.zeros(v1.shape, dtype=v1.dtype) for k1, v1 in v0.items()
            } for k0, v0 in self._observation_space.items()
        })
        return observation

    def step(self, action):
        """
        RETURNS:
            observation (ObsType) 
                - An element of the environment's _observation_space as the next 
                observation due to the agent actions. 
            reward (SupportsFloat)
                - The reward as a result of taking the action.
            terminated (bool) 
                - Whether the agent reaches the terminal state 
            truncated (bool) 
                - Whether the truncation condition outside the scope of the MDP 
                is satisfied. Always  false
            info (dict) 
                - Contains auxiliary diagnostic information (helpful for 
                debugging, learning, and logging). 
            done (bool) 
                - (Deprecated) A boolean value for if the episode has ended. Always false
    """
        print('==='*30)
        print(action)
        for i, ele, lo, hi in zip(range(action.shape[0]), action, self.action_space.low, self.action_space.high):
            if not np.isinf(hi) and not np.isinf(lo):
                action[i] = lo+(((ele+1)/2)*(hi-lo))
            elif np.isinf(hi) and not np.isinf(lo):
                action[i] = lo+np.log(ele+2)
            elif not np.isinf(hi) and np.isinf(lo):
                action[i] = hi-np.log(ele+2)
            else:
                action[i] = ele*100
            print(i, ele, lo, action[i], hi)
        print('-=#=-'*18)
        action = unflatten(self._action_space, action)
        print(action)
        observation = self._empty_obs()
        gp_reward = 0
        if (action['actions'][0] or True) and self.last_result:
            print('@'*80)
            if self.knowledge_table['size'].sum()+self.last_result['size']-self.knowledge_table.iloc[action['store_last']]['size'] <= self.max_knowledge_nodes:
                self.last_result['tree'] = self.last_result['tree'].copy_out(self.knowledge_base)
                self.last_result['exists'] = True
                self.knowledge_table.iloc[action['store_last']] = self.last_result
                self.last_result = {}
        if (action['actions'][1] or True):
            print('$')
            gprh = action['gp_hyperparams']['ranged_hyperparams']
            crossover_rate, mutation_rate, mutation_sd, max_size, max_depth, elitism = gprh
            pop = self.max_thinking_nodes//max_size
            print('max_think_nodes:', self.max_thinking_nodes)
            print('max size:', max_size)
            print('pop:', pop)
            gp = self.gp_system(
                crossover_rate=crossover_rate,
                mutation_rate=mutation_rate,
                mutation_sd=mutation_sd,
                max_size=max_size,
                max_depth=max_depth
            )
            observatory = self.world.act(action['observation_params'])
            scoreboard = self.scoreboard_factory.act(
                action['gp_hyperparams']['fitness_weights'],
                observatory, 
                max_size=max_size,
                max_depth=max_depth,
                dv=self.world.dv
            )
            tree_factories = [
                f.act(p, treebank=gp) for f, p in zip(
                    self.tree_factory_factories, 
                    action['gp_hyperparams']['factory_params']
                )
            ]
            if 'factory_weights' in action:
                tree_factory = CompositeTreeFactory(gp, tree_factories, action['gp_hyperparams']['factory_weights'])
            else:
                tree_factory = tree_factories[0]
            gp.operators = tree_factory.operators
            record, best_scores, best_tree = gp.run( # ____________/____________________
                tree_factory, # === === === === tree+factory __/_\/_____________________
                observatory,  # === === === === Observatory _\/____/____________________
                self.gp_steps, # == === === === steps __/________\/__Make this an action param later, leave for now
                int(pop), # === === === === === pop __\/___/____________________________
                self.world.dv, # == === === === dv ______\/_________________________/___
                fitness=scoreboard, # = === === fitness, def_fitness, temp_coeff _\/__/_
                elitism=int(elitism), # === === elitism ____________________________\/__
                ping_freq=1
            )
            observation['gp_run'] = {}
            for k in self.obs_measures:
                observation['gp_run'][k] = np.array(record[k], dtype=self._observation_space['gp_run'][k].dtype) 
            gp_reward = best_scores[self.gp_reward_measure]
            for i, name in enumerate(self.world.action_param_names):
                best_scores[name] = action['observation_params'][i]
            best_scores['tree'] = best_tree.copy_out()
            self.last_result = best_scores
        knowledge_reward = self.knowledge_table[
            self.knowledge_table['exists']][self.gp_reward_measure].mean()
        knowledge_reward = 0 if np.isnan(knowledge_reward) else knowledge_reward
        reward = gp_reward+knowledge_reward
        observation['knowledge'] = {
            k: np.array(col, dtype=self._observation_space['knowledge'][k].dtype) 
            for k, col 
            in self.knowledge_table.items() 
            if k != 'tree'
        }
        self.knowledge_table.to_csv(f'knowledge_{self.num_actions}.csv')
        print(
            f"{self.num_actions}: knowledge score:\n{knowledge_reward}\n------\n" + 
            f"thinking score:\n{gp_reward}\n-------\ntotal\n{reward}"
        )
        self.num_actions += 1
        print("~~~"*20)
        print(observation)
        print("~~~"*40)
        return flatten(self._observation_space, observation), reward, self.num_actions < self.max_actions, False, {'asdf': 'asdf'}, False


    def render(self):
        """Not needed"""
        return None

    def close():
        pass


    #----#----#---20    #----#----#---40    #----#----#---60    #----#----#---80
# class PhilosoPyAgent(PyEnvironment):
#     """Note that `PhilosoPyAgent` is seen as an 'environment' by the
#     `PhilosoPyRLAgent`, which is the RL subsystem of `PhilosoPyAgent`,
#     in charge of higher-level decisions like when to make observations,
#     run GP, publish results, etc.
#     """

#     @dataclass
#     class GPMemory:
#         observation_params: dict
#         observations: list
#         treebank: GPTreebank

#     def __init__(
#             self, 
#             world: World,
#             operators: list[ops.Operator],
#             tree_factories: list[TreeFactory],
#             name: str
#         ):
#         self.world = world
#         self.name = name
#         self.action_spec = [
#             world.observation_params().replace(name='observation'),
#             world.observation_params().replace(name='train_gp_static'),
#             world.observation_params().replace(name='train_gp_live'),
#             world.observation_params().replace(name='test_gp_predict'),
#             BoundedArraySpec(name='publish_bet'), # don't implement until ready for multi-agent
#             BoundedArraySpec(name='take_bet'), # don't implement until ready for multi-agent
#             BoundedArraySpec(name='read_repo'), # don't implement until ready for multi-agent
#             BoundedArraySpec(name='add_gp_result_to_knowledge'), 
#             BoundedArraySpec(name='add_repo_result_to_knowledge'), # don't implement until ready for multi-agent
#             BoundedArraySpec(name='prune_knowledge'),
#             BoundedArraySpec(name='prune_records')
#         ] #: tf_agents.typing.types.NestedTensorSpec,
#         self.knowledge = GPTreebank(operators=operators)
#         # structure this - maybe make a dataclass?
#         self.records: list[PhilosoPyAgent.GPMemory] = []

#     def time_step_spec(self):
#         """Return time_step_spec."""
#         pass

#     def observation_spec(self):
#         """Return observation_spec."""
#         pass

#     def action_spec(self):
#         """Return action_spec."""
#         return self.action_spec

#     def _reset(self):
#         """Return initial_time_step."""
#         pass

#     def _step(self, action):
#         """Apply action and return new time_step."""
#         pass

# class PhilosoPyRLAgent(ppo_agent.PPOAgent):
#     def __init__(
#             self, 
#             environment: PhilosoPyAgent
#         ):
#         first_step = environment.step()
#         super(PhilosoPyRLAgent, self).__init__(
#             time_step_spec = first_step, #: tf_agents.trajectories.TimeStep,
#             action_spec = environment.action_spec()
#             # optimizer = '???', #: Optional[types.Optimizer] = None,
#             # actor_net = '???', #: Optional[tf_agents.networks.Network] = None,
#             # value_net = '???' #: Optional[tf_agents.networks.Network] = None
#         )

print('OK')


        # reminder="""
        #     # 1: params for RUN_GP 
        #     # 1.1: gp hyperparams
        #     ---crossover_rate: float, [0, 1]
        #     ---mutation_rate: float, [0, 1]
        #     ---mutation_sd: float, [0, inf?]
        #     ---temp_coeff: float, [0, inf]
        #     ---max_depth: int, [1,100] user def
        #     ---max_size: int, [1,1000] user def
        #     ---tree_factory_weights: list[float], len=num factories, sum to 1
        #     ---tree_copy_protocols_weights: list[float], len=num protocols, sum to 1
        #     ---fitness_measure_weights: list[float], len=num measures, sum to 1
        #     ---tree factory params
        #     ------ laterparams for tree copy protocols (not needed with measures, I think)
        #     # 1:2 observation params (from World, defined for SineWorld case only, for now)
        #     ---start: float,
        #     ---end: float,
        #     ---n: int,
        #     x_rand: bool
        # """