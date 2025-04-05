from repository import Publication
from world import World, SineWorld
#from observatories import SineWorldObservatoryFactory
from tree_factories import RandomPolynomialFactory, RandomTreeFactory, RandomAlgebraicTreeFactory, SimpleRandomAlgebraicTreeFactory
from ppo import ActorCriticNetworkTanh, ActorCriticNetwork
from agent_controller import AgentController
from agent import Agent
from gp import GPTreebank
from gp_fitness import SimpleGPScoreboardFactory, SimplerGPScoreboardFactory
from model_time import ModelTime
from reward import Reward, Curiosity, Renoun, GuardrailCollisions
from tree_funcs import sum_all
from sb_statfuncs import mean, mode, std, nanage, infage, Quantile
from mutators import single_leaf_mutator_factory, single_xo_factory, random_mutator_factory
from hd import HierarchicalDict as HD
from utils import name_dict as nd

from typing import Container, Callable
from collections.abc import Sequence
import asyncio
from pathlib import Path
from copy import copy, deepcopy
from datetime import datetime

# from icecream import ic

import numpy as np
import pandas as pd

import json
import re

class Model:
    def __init__(self,
        world: World,
        dancing_chaos_at_the_heart_of_the_world: np.random.Generator,
        agents: Container[Agent],
        publications: Publication=None,
        sb_factory: SimpleGPScoreboardFactory=None,
        #rewards: list[Reward]=None,
        time: ModelTime=None,
        ping_freq=10, 
        out_dir=None,
    ):
        self.world = world
        self.agents: Container[Agent] = agents
        # XXX XXX XXX XXX does this VVV do anything?
        for agent in agents:
            agent.model = self
        self.agent_name_set = set([agent.name for agent in self.agents])
        self.publications = publications
        self.rng = dancing_chaos_at_the_heart_of_the_world
        self.sb_factory = sb_factory
        self.rewards = []
        self.t = time
        self.out_dir = out_dir if out_dir is not None else Path("")
        
    @property
    def json(self)->dict:
        """Outputs a json representation of the initial parameters of the
        Model, so that it can be recreated from the json. This includes
        generating json for objects of other philoso.py classes, which are
        passed into the `Model.__init__` method

        >>> # test me 
        """
        # `Model` contains `Agent` objects, with `AgentControllers`; 
        # the output JSON can contain parameters for single agents,
        # or populations of agents that start with the same paraneters.
        # Standardly, agents in a population will have a common naming 
        # prefix, `prefix`, and a number, such that agents are numbered
        # from 0 to n-1. The name of the agent, then, will be `prefix`_`idx`.
        # However, the Model itself does not necessarily explicitly 
        # represent Agents as belonging to populations, so this code 
        # identifies populations, which can be represented with a single 
        # JSON template, and single agents, which need a JSON 
        # representation of their own.
        # -------
        # `pop_max_idx` records the highest observed `idx` value for each
        # population. This should be equal to the population size, minus 1  
        pop_max_idxs = {}
        pop_sizes = {}
        # `pop_idxs` records the observed `idx` values for a population,
        # and is only actually used if it isn't just all the ints in
        # `range(n)` where `n` is the population size
        pop_idxs = {}
        # `nonpop_ag_names` records names of Agents not included in
        # populations: if there are any such agents, this is directly
        # included in the output JSON
        nonpop_ag_names = [] 
        # records the names of populations - the prefixes in agent names.
        # If there are any populations of `n` greater than 1, this is
        # included in the output JSON 
        pop_names = []
        # The actual JSON representations of Agent objects, which will be
        # included in the output JSON 
        agent_templates = {}
        # If any Agents at first appear to belong to a population, but 
        # then turn out not to be representable with a single JSON
        # template, the population will be split into single Agents.
        # This list records the populations for which this is done 
        split_pops = []
        # Loop over all agents...
        for a in self.agents:
            # ... and use their own .json property to JSONise them
            agent_json = a.json
            # Agents have `name` and `prefix` properties ... 
            a_name = agent_json['controller']['name']
            # ... but if the AgentController has no `prefix`, it will 
            # just use `name` for both. ...
            a_prefix = a.prefix
            idx = -1
            if a_name==a_prefix:
                # ... However, if the `{prefix}_{idx}` format is 
                # followed, we can split the name to get the prefix
                if match := re.fullmatch(r'(.*)_([0-9]+)', a_name):
                    agent_json['controller']['prefix'] = a_prefix = match[1]
                    idx = int(match[2])
            # If at this point the name and prefix are still the same, 
            # this Agent must be singly represented as not belonging to 
            # a population
            if a_name==a_prefix:
                nonpop_ag_names.append(a_name)
                agent_templates[a_name] = agent_json
            else:
                # Use regex to get idx, if it hasn't already been got
                if idx < 0:
                    idx = int(re.fullmatch(r'(.*)_([0-9]+)', a_name)[2])
                # The first disjunct indicates that this branch is for 
                # Agents that are the first of their population to be
                # recorded, since in this case the JSON template has yet
                # to be added to `json_templates`. This also applies in 
                # the case of non-population Agents, for which prefix
                # and name are identical, as these must be individually
                # recorded. 
                # However, ...
                if (a_prefix not in pop_names) or (a_prefix in split_pops):
                    # ...it's ALSO for Agents that appear to be in
                    # populations, but that population has been split into 
                    # non-population Agents, because they don't all have 
                    # the same template: In this case, `name` is used
                    # instead of `prefix`. This is the purpose of the 
                    # second disjunct above, and is why we need `id`
                    # below to behave differently in these two cases.
                    if a_prefix in split_pops:
                        id = a_name 
                        # In this case, the id is the name of an Agent that 
                        # does not belong to a population
                        nonpop_ag_names.append(id)
                    else: 
                        id = a_prefix
                        # In this case, the ID is a population name
                        pop_names.append(id)
                    # Add the template to the JSON
                    agent_templates[id] = agent_json
                # if the new agent IS apparently a member of a population,
                # AND that population has already had a template recorded
                # in `agent_templates`, we need to check subsequent observed
                # agents, to ensure that they *do* use the same template.
                # The comparison is made using HierarchicalDict.eq_except
                # because we *do* expect that the output JSON will include
                # the Agent's `name`, which will be different for each 
                # Agent even within a population, so the comparison should
                # exclude this 
                elif not HD(agent_json).compare_except(agent_templates[a_prefix], ['controller', 'name'], 'name'):
                    # Uh-oh, the Agent appeared to be a population member, 
                    # but it differs from the others in the same 
                    # population. So, we treat is as non-population,
                    # but also we need to do the same with previous and
                    # subsequent members of the same apparent population
                    # -----
                    # Record the template, using the agent name ...
                    agent_templates[a_name] = agent_json
                    # ... and record the name as a `nonpop_ag_name`
                    nonpop_ag_names.append(a_name)
                    # ... but now all the agents already recorded as members of 
                    # populations have to be individually recorded as non-pop
                    # agents. There's maybe more elegant ways of going about this,
                    # but this will do for now b=ouo
                    for idx in pop_idxs[a_prefix]:
                        # Set them all up with individual templates...
                        agent_templates[f'{a_prefix}_{idx}'] = deepcopy(agent_templates[a_prefix])
                        # And add them to the nonpop names
                        nonpop_ag_names.append(f'{a_prefix}_{idx}')
                    # then clear out all the records of them as members of a population 
                    del agent_templates[a_prefix]
                    del pop_sizes[a_prefix]
                    del pop_max_idxs[a_prefix]
                    del pop_idxs[a_prefix]
                    # Note that this is a former population that has 
                    # been split
                    split_pops.append(a_prefix)
                    # and remove from the list of populations
                    pop_names.remove(a_prefix)
                # Now, some bookkeeping. Update the max index if the
                # current index is the highest seen so far for its prefix... 
                pop_max_idxs[a_prefix] = max(idx, pop_max_idxs.get(a_prefix, 0))
                # ... increment the population size ...
                pop_sizes[a_prefix] = pop_sizes.get(a_prefix, 0) + 1
                # ... and add the index to the list of indices.
                pop_idxs[a_prefix] = pop_idxs.get(a_prefix, []) + [idx]
        for p in pop_names:
            # The max index for each population should be one less than
            # the pop size. If not, make individual JSONs for each agent
            # in the population, and treat them as non-population agents 
            if pop_max_idxs[p]+1 != pop_sizes[p]:
                for idx in pop_idxs[p]:
                    agent_templates[f'{p}_{idx}'] = deepcopy(agent_templates[p])
                    nonpop_ag_names.append(f'{p}_{idx}')
                del agent_templates[p]
                del pop_sizes[p]
                del pop_max_idxs[p]
                del pop_idxs[p]
                pop_names.remove(p)
        # Each template needs to show the population size. For templates
        # that apply to multiple agents, there should not be a 'name'
        # field 
        for pref, tmpt in agent_templates.items():
            tmpt['n'] = pop_sizes.get(pref, 1)
            if tmpt['n'] > 1:
                del tmpt['controller']['name']
        # Besides the agents, it is also necessary to return a number
        # of other params that are needed to recreate the model. These
        # are either available as object attributes of the Model, or
        # of other objects which are attrs of the Model. The 
        # `HierarchicalDict` class allows `list`s of keys to be used
        # to address dicts within dicts, and lower-level dicts inherit
        # key-value pairs from higher-level ones, unless they have
        # overriding key-value pairs. The `HD.simplify()` call moves
        # pairs into lower-level dicts, consolidating any identical
        # pairs in different dicts on different branches to their
        # nearest common parent, and removes any which are redundant
        return HD({
            "seed": self.rng.bit_generator.seed_seq.entropy,
            "out_dir": str(self.out_dir),
            "world": self.world.__class__.__name__,
            "world_params": self.world.json,
            "sb_factory": self.sb_factory.__class__.__name__,
            "sb_factory_params": self.sb_factory.json,
            "publication_params": self.publications.json,
            "agent_populations": pop_names,
            'agent_names': nonpop_ag_names,
            "agent_templates": agent_templates,
            "rewards": [rw.__name__ for rw in self.rewards],
            "reward_params": {rw.__name__: rw.json for rw in self.rewards}
        }).simplify()

    def add_reward(self, rew: Reward):
        self.rewards.append(rew)

    def run(self, days, steps_per_day, state_file:str=None, prefix=''): 
        if not self.rewards:
            raise AttributeError("model has no agent rewards")
        if state_file: # loading state a nice thing for later
            pass
        self.path_ = Path(self.out_dir)
        self.path_.mkdir(parents=True, exist_ok=True)
        for day in range(days):
            print(f'=== DAY {day} ===')
            asyncio.run(self.day(steps_per_day, day)) 
            self.night()
        # Generate model outputs
        for r in self.rewards:
            r.record.to_parquet(self.path_ / f'{prefix}{r.__name__}_record.parquet')
        for i, table in enumerate(self.publications.tables):
            table['tree'] = table['tree'].apply(lambda x: f"{x}")
            table.to_parquet(self.path_ / f'{prefix}publication_{i}_end.parquet')
        for agent in self.agents:
            for i, table in enumerate(agent.ac.memory.tables):
                table['tree'] = table['tree'].apply(lambda x: f"{x}")
                table.to_parquet(self.path_ / f'{prefix}{agent.name}_mem_{i}.parquet')
            agent.save_nn(f'{self.path_}/{prefix}{agent.name}_nn_state.pt')
        pd.DataFrame({
            agent.name: agent.day_rewards for agent in self.agents
        }).to_parquet(self.path_ / f'{prefix}day_rewards.parquet')
        print("The model is done. Goodbye!")


    async def day(self, steps, day):
        print('Good morning!')
        for agent in self.agents:
            agent.morning_routine(steps)
        print('Oh, what a lovely day')
        for step in range(steps):
            self.not_done = self.agent_name_set.copy()
            self._reward_dict = {}
            shuffled_agents = copy(self.agents)
            self.rng.shuffle(shuffled_agents)
            tasks = [asyncio.create_task(agent.day_step()) for agent in shuffled_agents]
            _ = await asyncio.wait(tasks)
            for a in self.agents:
                for i, tbl in enumerate(a.ac.memory.tables):
                    for j, row in tbl.iterrows():
                        if row['exists'] and ((row['tree'].size() != row['size']) or (row['tree'].depth() != row['depth'])):
                            raise ValueError(
                                f"At time {self.t}, "+
                                f"agent {a.name} has tree {row['tree']} recorded at table "+
                                f"{i}, row {j} with size"+
                                f" and depth {row['size'], row['depth']} when it should be"+
                                f" {row['tree'].size(), row['tree'].depth()}"
                            )
                        if row['exists'] and row['size']==1:
                            raise ValueError(
                                f"At time {self.t}, "+
                                f"agent {a.name} has tree {row['tree']} recorded at table "+
                                f"{i}, row {j} with size and depth 1. Huhhhhhhh?"
                            )
            for i, tbl in enumerate(self.publications.tables):
                for j, row in tbl.iterrows():
                    if row['exists'] and ((row['tree'].size() != row['size']) or (row['tree'].depth() != row['depth'])):
                        raise ValueError(
                            f"At time {self.t}, "+
                            f"publication {i} has tree {row['tree']} recorded at"+
                            f" row {j} with size"+
                            f" and depth {row['size'], row['depth']} when it should be"+
                            f" {row['tree'].size(), row['tree'].depth()}"
                        )
                    if row['exists'] and row['size']==1:
                        raise ValueError(
                            f"At time {self.t}, "+
                            f"agent {a.name} has tree {row['tree']} recorded at table "+
                            f"{i}, row {j} with size and depth 1. Huhhhhhhh?"
                        )
            self.t.tick()
        self.daily_journals(day)
        print('What a beautiful sunset!')
        for agent in self.agents:
            agent.evening_routine()

    def daily_journals(self, day):
        for a in self.agents:
            base: Path = self.path_ / a.name / 'days'
            base.mkdir(parents=True, exist_ok=True)
            a.ac.memory.save(
                base / f'day_{day}_mems', 
                'parquet'
            )
            a.save_training_buffer(
                base / f'day_{day}_actions.csv'
            )
        pubdir = self.path_ / 'publications'
        pubdir.mkdir(parents=True, exist_ok=True)
        self.publications.save(
            pubdir / f'day_{day}_jrnl',
            'parquet'
        )

    def night(self):
        print("Good night!")
        for agent in self.agents:
            print(f'Agent {agent} is dreaming')
            agent.night()
            print(f'Agent {agent} is sleeping deeply')

    def mark_done(self, name):
        self.not_done -= {name}

    async def get_rewards(self, name):
        while self.not_done:
            await asyncio.sleep(0.2)
        if not self._reward_dict:
            self._reward_dict = sum_all(*[reward() for reward in self.rewards])
        return self._reward_dict[name]


def example_model(seed: int=None, out_dir: str|Path=Path('output', 'test'), ping_freq=5) -> Model:
    dancing_chaos_at_the_heart_of_the_world = np.random.Generator(np.random.PCG64(seed))
    world = SineWorld(
        50, 100, 0.05, (10,100), (0.1, 1), 
        seed=dancing_chaos_at_the_heart_of_the_world
    )
    print(f'Seed: {dancing_chaos_at_the_heart_of_the_world.bit_generator.seed_seq.entropy}')
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
    agent_indices = {f'ag_{i}': i for i in range(n_agents)}
    pub = Publication(
        gp_vars_core + gp_vars_more, # cols: Sequence[str],
        10, # rows: int,
        time, # model_time: ModelTime,
        agent_indices,
        types = np.float64, # types: Sequence[dtype] | Mapping[str, dtype] | dtype | None = None,
        tables = 2, # tables: int = 1,
        reward = 'ranked', # reward: PublicationRewardFunc | str | None = None,
        value = 'value'
        # DEFAULTS USED decay: float = 0.95, value: str = "fitness",
    )
    agents = [
        Agent(
            AgentController(
                world, # World, (is now the ObservatoryFactory)
                time, # ModelTime,
                name, # name,
                6, # mem_rows,
                3, # mem_tables,
                world.dv, # dv,
                'irmse', # str, def_fitness
                sb_factory, # SimpleGPScoreboardFactory, # Needs to be more general XXX TODO
                [SimpleRandomAlgebraicTreeFactory], #tree_factory_classes, # tree_factory_classes: list[type[TreeFactory]],
                dancing_chaos_at_the_heart_of_the_world, # np.random.Generator,
                agent_indices, #agent_names, # dict[str, int],
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
        ) for name in agent_indices.keys()
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
        out_dir=out_dir
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
    return model

class ModelFactory:
    """A class used to generate and run philoso.py Models from JSON
    files or objects. This could have been done statically, with
    class methods or functionally, but this approach was chosen so users
    can add custom Worlds, networks, mutators, etc. Custom classes
    and functions are added when the initialiser is run, and 
    then multiple Models can be run from the same ModelFactory.

    Note that all the Attributes listed below are class attributes,
    but are overriden by instance attributes of the same name on
    initialisation. The instance attributes comprise the same values
    as the class attributes, plus any values added on initialisation
    via the parameters of the same name
    
    Parameters
    ----------
    worlds : Sequence[class[World]]
        Custom `World` classes to be added to the `worlds` attribute
    sb_factories : Sequence[class[GPScoreboardFactory]]
        Custom `GPScoreboardFactory` classes to be added to the 
        `sb_factories` attribute
    network_classes : Sequence[class[torch.Module]]
        Custom neural network classes to be added to the
        `network_classes` attribute
    tree_factory_classes : Sequence[class[TreeFactory]]
        Custom `TreeFactory` classes to be added to the 
        `tree_factory_classes` attribute
    mutators : Sequence[Callable]
        Custom `mutator` functions to be added to the `mutators`
        attribute
    gp_systems : Sequence[Treebank]
        Custom GP Systems classes to be added to the `gp_systems` 
        attribute
    rewards : Sequence[Reward]
        Custom `Reward` classes to be added to the `rewards` attribute

    Attributes
    ----------
    worlds : dict[str, class[World]]
        `dict` to allow JSON data to specify which `World` class (which
        can be a provided class, or custom) should be used in a `Model`
    sb_factories : dict[str, class[GPScoreboardFactory]]
        `dict` to allow JSON data to specify which `GPScoreboardFactory` 
        class (whichcan be a provided class, or custom) should be used 
        in a `Model`, as part of the Genetic Programming system. A 
        scoreboard includes a pipeline of functions that calculate 
        measures of model performance, or other measures of 
        characteristics of models (trees) generated by GP. These are
        mainly used to calculate fitness, but are also useful for allowing
        the RL system to observe characteristics of the GP outputs
    network_classes : dict[str, class[torch.Module]]
        `dict` to allow JSON data to specify which pytorch Module class 
        should be used in a `Model`, to provide the Neural Networks
        that control the `Agents`
    tree_factory_classes : dict[str, class[TreeFactory]]
        `dict` to allow JSON data to specify which `TreeFactory` class 
        (which can be a provided class, or custom) should be used in a 
        `Model` to initialise the GP system with random trees
    mutators : dict[str, Callable]
        `dict` to allow JSON data to specify which mutator functions 
        (whichcan be provided or custom) should be used in a `Model`'s
        GP system to mutate trees when they are copied
    gp_systems : dict[str, Treebank]
        `dict` to allow JSON data to specify which GP system (provided 
        or custom) should be used in by `Model`'s `Agents` to form 
        generalisations over their `World`
    rewards : dict[str, Reward]
        `dict` to allow JSON data to specify which `Reward` classes
        generate rewards for `Agent`'s RL systems, based on the state 
        of the model as a whole
    """

    # `nd` is a function that turns a collection of objects (classes
    # or functions) with a `__name__` attr into a dictionary in which 
    # the `__name__`s are the keys and the objects are the values 
    worlds = nd(SineWorld)
    sb_factories = nd(
        SimpleGPScoreboardFactory, 
        SimplerGPScoreboardFactory
    )
    network_classes = nd(
        ActorCriticNetworkTanh,
        ActorCriticNetwork
    )
    tree_factory_classes = nd(
        SimpleRandomAlgebraicTreeFactory
    )
    mutators = nd(
        single_leaf_mutator_factory, 
        single_xo_factory, 
        random_mutator_factory
    )
    gp_systems = nd(GPTreebank)
    sb_statfuncs = nd(mean, mode, std, Quantile, nanage, infage)
        # The `Observation` which uses these statfuncs ignores all `nan`,
        # `inf`, and `-inf` values, so it is useful to also note how many
        # of these values there are  
    rewards = nd(Curiosity, Renoun, GuardrailCollisions)

    def __init__(self, 
            worlds:Sequence=None, 
            sb_factories:Sequence=None, 
            network_classes:Sequence=None, 
            tree_factory_classes:Sequence=None,
            mutators:Sequence=None,
            gp_systems:Sequence=None,
            sb_statfuncs:Sequence=None,
            rewards:Sequence=None
        ):
        # creates instance attributes out of all the class attribute
        # dicts, with any values passed to the initialiser params added
        # also 
        self.worlds = {**self.worlds, **nd(*worlds or [])}
        # `*param or []` unrolls `param` if param is not `None`, 
        # otherwise unrolls *[]. `nd` takes its *args and makes a dict,
        # such that each `arg` is a value, and `arg.__name__` as the key.
        # `{**dic1, **dic2}` creates a combined dictionary with the keys
        # and values of dic1 and dic2, with dic2 overriding dic1 in
        # the case of key collisions 
        self.sb_factories = {**self.sb_factories, **nd(*sb_factories or [])}
        self.network_classes = {**self.network_classes, **nd(*network_classes or [])}
        self.tree_factory_classes = {**self.tree_factory_classes, **nd(*tree_factory_classes or [])}
        self.mutators = {**self.mutators, **nd(*mutators or [])}
        self.gp_systems = {**self.gp_systems, **nd(*gp_systems or [])}
        self.sb_statfuncs = {**self.sb_statfuncs, **nd(*sb_statfuncs or [])}
        self.rewards = {**self.rewards, **nd(*rewards or [])}
    
    def _read_json(self, json_: dict|str) -> HD:
        """`ModelFactory` represents json data, not using the standard
        built-in `dict` class, but a custom `dict` subclass, 
        `HierarchicalDict`, (`HD`) in which `dicts` nested in `dicts`inherit 
        or override key-value pairs in the parent `dict`, and lists
        of keys can be passed as keys, such that:
        
        ```
        dic[['a', 'b', 'c']]
        ```
        
        is syntactic sugar for:

        ```
        dic['a']['b']['c']
        ```

        However, a valid input can be:
        
        1) a valid json string
        2) a file address for a valid json file
        3) a json-formatted python `dict`
        4) a json-formatted HierarchicalDict

        This method takes any of (1) to (4) and outputs (4)

        Parameters
        ----------

        json_ : dict|HierarchicalDict|str
            JSON data (in `str`, `dict`, or `HierarchicalDict` formats) 
            or a str giving the location of a JSON file to provide all 
            the parameters needed to create a philoso.py Model

        Returns
        -------
        HierarchicalDict
            A HierarchicalDict of the `json_` data.

        """
        if not isinstance(json_, HD):
            if isinstance(json_, str):
                if json_.strip()[0] in '[{':
                    json_ = json.loads(json_)
                with open(json_) as f:
                    json_ = json.load(f)
            json_ = HD(json_)
        return json_

    def from_json(self, json_: str|dict, out_dir: str=None) -> Model:
        """This represents the core functionality of the ModelFactory
        class: it creates a ready-to-run philoso.py `Model` from a 
        JSON input.

        Parameters
        ----------

        json_ : dict|HierarchicalDict|str
            JSON data (in `str`, `dict`, or `HierarchicalDict` formats) 
            or a str giving the location of a JSON file to provide all 
            the parameters needed to create a philoso.py Model

        Returns
        -------
        Model
        """
        # ensure the json data is in HierarchicalDict format
        json_ = self._read_json(json_)
        # seed the random number generator with seed provided in json,
        # or let numpy generate its own seed, if no seed value is 
        # provided. The seed is printed for reproducibility.
        seed = json_.get('seed', None)
        match seed:
            case 42:
                print("DON'T PANIC")
            case 69:
                print("NICE")
            case 420:
                print(
                    "BLAZE IT. Actually don't, at least not while you're coding. " +
                    "Or do. I don't care. I'm not your mum."
                )
            case 666:
                print("HAIL SATAN")
        dancing_chaos_at_the_heart_of_the_world = np.random.Generator(
            np.random.PCG64(json_.get('seed', None))
        )
        print(f'Seed: {dancing_chaos_at_the_heart_of_the_world.bit_generator.seed_seq.entropy}')
        # `World` has a static method `from_json` which locates the branch
        # of the json HierarchicalDict that contains the params for 
        # creating a `World` instance, and creates an instance with those
        # params
        world = self.worlds[json_['world']].from_json(json_)
        # So does ScoreboardFactory
        sb_factory = self.sb_factories[json_['sb_factory']].from_json(json_)
        # ModelTime acts as a timekeeper for the model, and can be
        # created with no params
        time = ModelTime()
        # the agent_names key in json holds a list of names of agents
        # that do not belong to a population (a population being two 
        # or more Agents spawned using the a single json template)
        listed_names = json_.get('agent_names', [])
        # We're going to make a list of all the agent names, including
        # those that belong to populations - so rather than add the
        # population-members to the list in `_json`, they are added to a 
        # copy, which is made below 
        agent_name_list = list(listed_names)
        # Iterate through the names of agent populations, which are
        # listed in `agent_populations`, to add the names of all 
        # agents in each population  
        for nom in json_[['agent_populations']]:
            # For a population to have agents, the population template
            # must include a positive value `n`, which gives the 
            # population size. If `n` is greater than 1...
            if json_.get(["agent_templates", nom, "n"], 0) > 1:
                # ... add each agent in the population's name to the
                # agent_name_list as: 
                # '{population name}_{0}' ... '{population name}_{n-1}'
                agent_name_list += [
                    f'{nom}_{i}' for i in range(
                        json_[["agent_templates", nom, "n"]]
                    )
                ]
            else:
                # Otherwise, just add the population name, treating
                # it as also being the name of the only agent in that
                # population 
                agent_name_list.append(nom)
        # If the list has more members than `set(list)`, there are 
        # duplicates ...
        if len(set(agent_name_list)) < len(agent_name_list):
            # So check which names are duplicated
            for i, nm in enumerate(agent_name_list):
                if nm in agent_name_list[i+1:]:
                    raise ValueError(
                        f"Duplicate name: {nm} in {agent_name_list}"
                    )
        # A couple of the helper classes need dicts that reverse the map
        # of agent names to indices in the agent list - e.g. Publication, 
        # which is created next 
        agent_names_2_idxs = {name: i for i, name in enumerate(agent_name_list)}
        # Publication is another class that can be created from JSON
        pub = Publication.from_json(json_, time=time, agent_names=agent_names_2_idxs)
        # Cool, now it's time to make the actual Agents! First, an
        # empty list to store them in 
        agents = []
        # JSON should contain either a list of individual Agent names,
        # or of population names, or both. Iterate through both lists
        # to create the needed Agents 
        for agdata in ["agent_names", "agent_populations"]:
            for nom in json_.get([agdata], []):
                if agdata=="agent_populations" and nom in listed_names:
                    raise ValueError(
                        f"{nom} cannot be used as both an agent name and a" +
                        "population prefix"
                    )
                # Agents require a number of helper classes and functions,
                # which here are picked out from the dictionaries stored
                # in ModelFactory, using data from json. Note the use of
                # lists of keys to address items in the nested dictionaries
                # of the json. Each agent needs...
                tree_factory_classes = [
                    self.tree_factory_classes[tfc]
                    for tfc 
                    in json_[[
                        "agent_templates", 
                        nom, 
                        "controller", 
                        "tree_factory_classes"
                    ]] # One or more tree factory classes
                ]
                gp_system = self.gp_systems[
                    json_[[
                        "agent_templates", 
                        nom, 
                        "controller", 
                        "gp_system"
                    ]] # One GP system
                ] 
                network_class = self.network_classes[
                    json_[[
                        "agent_templates", 
                        nom, 
                        "controller", 
                        "network_class"
                    ]] # And one network class. 
                ] # Other needed helpers are passed to the Agent or
                # AgentController from_json methods by passing the
                # whole dict from ModelFactory (e.g. self.mutators)
                # to the from_json call as a kwarg
                # ------------------------
                # Using a list comprehension to create as many agents as 
                # needed from each template. If no value for `n` is given,
                # `n` of 1 is assumed 
                agents += [
                    Agent.from_json(
                        json_,
                        AgentController.from_json(
                            json_, 
                            world=world,
                            time=time,
                            name=nom if nom in listed_names else f'{nom}_{i}', 
                            prefix=nom,
                            sb_factory=sb_factory, 
                            tree_factory_classes=tree_factory_classes,
                            agent_indices=agent_names_2_idxs,
                            repository=pub,
                            gp_system=gp_system,
                            mutators=self.mutators,
                            sb_statfuncs=self.sb_statfuncs,
                            rng=dancing_chaos_at_the_heart_of_the_world
                        ),
                        dancing_chaos_at_the_heart_of_the_world,
                        prefix=nom,
                        network_class=network_class,
                    ) for i in range(json_.get(['agent_templates', nom, 'n'], 1))
                ]
        # Note, networks must be made *after* all agents have been made,
        # as some params depend on knowing how many other agents there are
        for agent in agents:
            prefix_ = agent.prefix
            agent.make_networks(
                **json_[['agent_templates', prefix_, 'network_params']]
            )
        # Now, the model itself can be created
        model = Model(
            world, #: World,
            dancing_chaos_at_the_heart_of_the_world, #: np.random.Generator,
            agents, #: Container[Agent],
            pub, # publications, #: Publication=None,
            sb_factory, #: SimpleGPScoreboardFactory=None,
            time, #: ModelTime=None
            out_dir=json_['out_dir']
            #json_['agent_populations']
        )
        # Each Agent needs a backreference to Model
        for agent in agents:
            agent.ac.model = model
        # Rewards need visibility into the model, so they are created
        # after `model`, and then are added to it.
        for reward in json_['rewards']:
            model.add_reward(
                self.rewards[reward].from_json(json_, model=model)
            )
        # DONE! MODEL!
        return model

    def save_json_and_run(self,
            model: Model, 
            steps_per_day: int,
            days: int,
            prefix='philoso_py_',
            json_out_dir=None,
            run=True):
        """ This method takes an existing model (which may or may
        not have been created by ModelFactory), makes it save a JSON
        file containing all the parameters needed to recreate it, and
        runs it. Parameters that are not added to any object's 
        initialiser, but which are used when calling `model.run`
        are included in this method's params, and are added to the
        JSON before it is saved and the Model is run.

        Parameters
        ----------

        model : Model
            The model to be saved and run
        steps_per_day : int
            A 'day' cycle generates Agent behaviour and its rewards,
            which is then used to train Agents networks during the 
            'night'. `steps_per_day` indicates the number of actions
            each Agent can take in one day
        days: int
            The number of days in the entire simulation
        prefix : str
            Prefix prepended to all output filenames for this 
            simulation run
        json_out_dir : str : optional
            Target directory for the saved JSON file. If no value
            is provided, the out_dir of the model is used
        run : bool :optional:True
            If False, the JSON is saved but the model is not run.
            If True, the model is run.
        """
        # Generate the JSON data using the model's `json` property,
        # and add in the `run` params, `days`, `steps_per_day`, and
        # `output_prefix` 
        saved_json = {
            **model.json,
            'days': days, 
            'steps_per_day': steps_per_day,
            'output_prefix': prefix
        }
        # If no json_out_dir is given, use the general file output
        # directory, or else the current directory 
        if json_out_dir is None:
            if "out_dir" in saved_json:
                json_out_dir = saved_json['out_dir']
            else:
                json_out_dir = ""
        # Create a Path object for this directory, and make sure it
        # exists 
        path = Path(json_out_dir)
        path.mkdir(parents=True, exist_ok=True)
        # append the json filename, to give the complete json filepath
        json_outpath = path / f'{prefix}params.json'
        # save the json to a file
        with open(json_outpath, 'w', encoding='utf-8') as f:
            json.dump(
                saved_json, f, 
                ensure_ascii=False, 
                indent=2
            )
        # Run the model if `run` is True
        if run:
            model.run(
                days, 
                steps_per_day,
                prefix=prefix
            )
    
    def run_json(self, json_: str|dict, out_dir: str=None):
        """This function creates and runs a model from a json file,
        with no arguments other than the json itself

        Parameters
        ----------
        json : str|dict
            A json dict or HierarchicalDict, a json str, or a str
            representing the location of a json file 
        """
        # Ensure the json is in the HierarchicalDict format
        json_ = self._read_json(json_)
        # If a target directory for outputs is given, prepend
        # this to 'out_dir' in json_
        if out_dir:
            json_['out_dir'] = out_dir + '/' + json_['out_dir'] 
        # Create the model
        model = self.from_json(json_)
        # retrieve the extra data needed to cal model.run
        days = json_['days']
        steps_per_day = json_['steps_per_day']
        output_prefix=json_['output_prefix']
        # Note the start time
        start = datetime.now()
        # this call creates a mew JSON file from the model,
        # and runs the model. Note it is still useful to save
        # json from the model, as it may have used default values
        # of params. These would not be in the original JSON, so
        # it's handy to keep a record 
        self.save_json_and_run(
            model, 
            steps_per_day,
            days,
            prefix=output_prefix)
        print('model ran for', str(datetime.now()-start))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='philoso-py')
    parser.add_argument('json_fn')
    parser.add_argument('-o', '--outdir')
    args = parser.parse_args()
    print('Running:', args.json_fn)
    kwargs = {'out_dir': args.outdir} if args.outdir else {}
    ModelFactory().run_json(args.json_fn, **kwargs)