from enum import Enum
from typing import Any, Sequence, Mapping, Callable
from pathlib import Path
from collections import OrderedDict
import numpy as np
#import pandas as pd
from world import World
from tree_factories import TreeFactory #, CompositeTreeFactory, TreeFactoryFactory
from gp import GPTreebank
from trees import Tree
from observatories import ObservatoryFactory
from guardrails import GuardrailManager
# import operators as ops
from dataclasses import dataclass
from world import World
from trees import Tree
from gymnasium import Env
from gymnasium.spaces import Dict #, Tuple, Discrete, Box, MultiBinary, MultiDiscrete, Space
from gymnasium.spaces.utils import flatten, flatten_space #, unflatten
# from gymnasium.utils import seeding
# from rl_bases import Actionable
from gp_fitness import SimpleGPScoreboardFactory
from repository import Archive, Publication
from model_time import ModelTime
from action import Action, GPNew, GPContinue, UseMem, StoreMem, Publish, Read
from observation import Observation, GPObservation, Remembering, LitReview
# from ppo import ActorCriticNetwork, PPOTrainer

from icecream import ic

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
    

class AgentController(Env):

    def __init__(self, 
            world: World,
            time: ModelTime,
            name: str,
            mem_rows: int,
            mem_tables: int,
            dv: str,
            def_fitness: str,
            sb_factory: SimpleGPScoreboardFactory, # Needs to be more general XXX TODO
            obs_factory: ObservatoryFactory,
            tree_factory_classes: list[type[TreeFactory]],
            rng: np.random.Generator,
            agent_names: dict[str, int],
            repository: Publication,
            out_dir: str|Path,
            record_obs_len: int,
            max_readings: int = 5,
            mem_col_types: Sequence[np.dtype]|Mapping[str, np.dtype]|np.dtype|None=None,
            gp_system=type[GPTreebank],
            sb_statfuncs: Sequence[Callable]=None,
            num_treebanks: int = 2,
            short_term_mem_size: int = 5,
            # fitness_measures = None,
            value: str="fitness", 
            # tree_factory_factories: list[TreeFactoryFactory]=None,
            # max_actions: int = 40, ## ??? XXX
            # gp_steps: int = 100,
            max_volume: int = 50_000,
            # max_thinking_nodes: int = 20_000,
            # max_knowledge_nodes: int = 1000,
            # max_knowledge_trees: int = 50,
            max_max_size: int = 400, ## ??? XXX
            max_max_depth: int = 100, ## ??? XXX
            theta: float = 0.05,
            gp_vars_core: list[str] = None,
            gp_vars_more: list[str] = None,
            # value_measures: list=None,
            # value_measure_weights: list[float]=None,
            device: str='cpu',
            guardrail_base_penalty = 1.0,
            ping_freq=5,
            *args, **kwargs
        ):
        """What should be in __init__, and what in reset?
        """
        # GLOBAL
        self.world = world
        self.model = None
        self.t = time
        self.np_random = rng # Note, this is a @property setter in Env
        self.name = name
        self.dv = dv
        self.out_dir = out_dir / self.name
        # GP CLASS SETUP
        self.gp_system = gp_system
        self.def_fitness = def_fitness
        self.max_volume = max_volume
        self.gptb_list = [None] * num_treebanks
        self.gptb_cts = [0] * num_treebanks
        self.sb_factory = sb_factory
        self.agent_names = agent_names
        self.theta = theta
        self.ping_freq = ping_freq
        self.short_term_mem_size = short_term_mem_size
        self.record_obs_len = record_obs_len
        self.guardrail_manager = GuardrailManager(base_penalty=guardrail_base_penalty)
        self._mems_to_use = []
        # self.tree_factory_factories = tree_factory_factories
        self.gp_vars_core = gp_vars_core if gp_vars_core else [ 
            'mse', 'rmse', 'size', 'depth', 'raw_fitness', 'fitness', 
        ]
        self.gp_vars_more = gp_vars_more if gp_vars_more else [
            'wt_fitness', 'wt_size', 'wt_depth', "crossover_rate", "mutation_rate", 
            "mutation_sd", "max_depth", "max_size", "temp_coeff", "pop", "elitism", 
            'obs_start', 'obs_stop', 'obs_num'
        ]
        self.sb_statfuncs = sb_statfuncs if sb_statfuncs else [
            # calculating std of a col of len 1 or 0 returns nan. The Guardrails
            # on `action.GPNew` and `action.GPContinue` should prevent Scoreboards
            # under this length, but filtering out infs and nans may result in 
            # short cols
            lambda col: col.mean() if len(col) > 0 else 0.0,
            lambda col: col.mode().mean() if len(col) > 0 else 0.0,
            # XXX TODO - there quantiles are acting weird
            *[(lambda col: col.quantile(q) if len(col) > 0 else 0.0) for q in np.linspace(0.0, 1.0, 9)],
            lambda col: col.std() if len(col) > 1 else 0.0,
            # The `Observation` which uses these statfuncs ignores all `nan`,
            # `inf`, and `-inf` values, so it is useful to also note how many
            # of these values there are  
            lambda col: col.isna().mean(),
            lambda col: np.isinf(col).mean()
        ]
        # MEMORY SET UP
        self.memory = Archive(
            cols       = self.gp_vars_out,
            rows       = mem_rows,  
            model_time = self.t,  
            types      = mem_col_types,  
            tables     = mem_tables,
            value      = value, 
            max_size   = max_max_size,
            max_depth  = max_max_depth,
            **kwargs
        )
        # READING SET UP
        self.max_readings = max_readings
        self.repository = repository
        self.repository._add_user(self)
        self.obs_factory = obs_factory
        self.tree_factory_classes = tree_factory_classes
    
    # ACTIONS
    def make_actions(self):
        self.actions: dict[str, Action] = {}
        self.actions["gp_new"]      = GPNew(self,
                                            self.obs_factory,
                                            self.tree_factory_classes,
                                            self.np_random,
                                            self.out_dir,
                                            self.t,
                                            self.dv,
                                            self.def_fitness,
                                            self.max_volume,
                                            self.theta,
                                            self.ping_freq
                                        )
        self.actions["gp_continue"] = GPContinue(self,
                                            self.out_dir,
                                            self.t
                                        )
        self.actions["use_mem"]     = UseMem(self) 
        self.actions["store_mem"]   = StoreMem(self) 
        self.actions["publish"]     = Publish(self)
        self.actions["read"]        = Read(self,
                                           max_readings=self.max_readings,
                                           vars=self.gp_vars_out
                                        )
        self._action_space = Dict(
            {k: v.action_space for k, v in self.actions.items()}
        )

    # OBSERVATIONS
    def make_observations(self):
        self.observations: dict[str, Observation] = {}
        self.observations['gp_obs']      = GPObservation(self,
                                                self.sb_statfuncs,
                                                self.record_obs_len
                                            )
        self.observations['remembering'] = Remembering(self)
        self.observations['lit_review']  = LitReview(self)
        self._observation_space = Dict(
            {k: v.observation_space for k, v in self.observations.items()}
        )


    @property
    def observation_space(self):
        return flatten_space(self._observation_space)

    @property
    def action_space(self):
        return self._action_space
    
    @property
    def gp_vars_out(self):
        return self.gp_vars_core + self.gp_vars_more

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
        if self.model is None:
            raise AttributeError('An AgentController must be assigned to a philoso_py.Model to run')
        # print(f'Seed: {self.np_random.bit_generator.seed_seq.entropy}')
        self.world.seed = self.np_random
        # for tff in self.tree_factory_factories:
        #     tff.seed = self.np_random
        
        # It is hardwired here that the default reward measure is imse, 
        # unless it is not calculated, then irmse, failing that isae
        # for fm in ['imse', 'irmse', 'isae']:
        #     if fm in self.fitness_measures:
        #         self.gp_reward_measure = fm
        #         break
        # else:
            # raise ValueError(
            #     f"Invalid value for fitness_measures {self.fitness_measures}: " +
            #     "at least one of 'imse', 'irmse', and 'isae' must be included" 
            # )
        return flatten(self._observation_space, self.observe()), {}
    
    # def _empty_obs(self):
    #     observation = OrderedDict({
    #         k0: {
    #             k1: np.zeros(v1.shape, dtype=v1.dtype) for k1, v1 in v0.items()
    #         } for k0, v0 in self._observation_space.items()
    #     })
    #     return observation
    
    def observe(self):
        return OrderedDict({k: obs() for k, obs in self.observations.items()})
        # ooo = OrderedDict({k: obs() for k, obs in self.observations.items()})
        # for k,v in ooo.items():
        #     ic('='*100)
        #     ic(k)
        #     ic(flatten(self.observations[k].observation_space, v).shape)
        #     if k == 'remembering':
        #         ic(self.observations[k].observation_space.sample().shape)
        #         ic(len(self.observations[k].gp_vars_core))
        #         ic(v.shape)
        #     if k == 'lit_review':
        #         vee=self.observations[k].observation_space.sample()
        #         ic(vee)
        #         ic(v)
        #         ic('---'*20)
        #         for vv1, vv2 in zip(v, vee):
        #             ic(vv1.shape)
        #             ic(vv2.shape)
        #             ic("_.-'-."*5)
        # return ooo

    @property
    def mems_to_use(self):
        return self._mems_to_use
    
    @mems_to_use.setter
    def mems_to_use(self, mems: Tree):
        mems = [mem for mem in mems if mem is not None]
        if len(mems) > self.short_term_mem_size:
            mems = mems[:self.short_term_mem_size]
        self._mems_to_use = mems

    # I think this needs to be async - must wait until all agents have acted before getting reward
    async def step(self, action): ### XXX XXX This needs to be broken the fuck up into constituent actions
        """
        RETURNS:
            observation (ObsType) 
                - An element of the environment's _observation_space as the next 
                observation due to the agent actions. 
            reward (SupportsFloat)
                - The reward as a result of taking the action.
            terminated (bool) 
                - Whether the agent reaches the terminal state: always false
            truncated (bool) 
                - Whether the truncation condition outside the scope of the MDP 
                is satisfied. Always  false
            info (dict) 
                - Contains auxiliary diagnostic information (helpful for 
                debugging, learning, and logging). 
            done (bool) 
                - (Deprecated) A boolean value for if the episode has ended. Always false
        """
        print(f'Agent {self.name} will attempt {list(action.keys())} at time {self.t}')
        for act, params in action.items():
            self.actions[act](params)
        print(f'Agent {self.name} has done {list(action.keys())} at time {self.t}')
        self.model.mark_done(self.name)
        reward = await self.model.get_rewards(self.name)
        observation = self.observe()
        return flatten(self._observation_space, observation), reward, False, False, {'asdf': 'asdf'}, False


    def render(self):
        """Not needed"""
        return None

    def close():
        pass




print('OK')

