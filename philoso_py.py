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

from icecream import ic

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
        out_dir=None
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
        pop_max_idxs = {}
        pop_sizes = {}
        pop_idxs = {}
        nonpop_ag_names = []
        pop_names = []
        agent_templates = {}
        split_pops = []
        for a in self.agents:
            agent_json = a.json
            a_name = agent_json['controller']['name']
            a_prefix = a.prefix
            if a_name==a_prefix:
                if match := re.fullmatch(r'(.*)_([0-9]+)', a_name):
                    agent_json['controller']['prefix'] = a_prefix = match[1]
            if a_name==a_prefix:
                nonpop_ag_names.append(a_name)
                agent_templates[a_name] = agent_json
            else:
                idx = int(a_name[len(a_prefix)+1:])
                if (a_prefix not in pop_names) or (a_prefix in split_pops):
                    pop_names.append(a_prefix)
                    agent_templates[a_prefix] = agent_json
                elif not HD(agent_json).eq_except(agent_templates[a_prefix], 'name'):
                    agent_templates[a_name] = agent_json
                    nonpop_ag_names.append(a_name)
                    for idx in pop_idxs[a_prefix]:
                        agent_templates[f'{a_prefix}_{idx}'] = deepcopy(agent_templates[a_prefix])
                        nonpop_ag_names.append(f'{a_prefix}_{idx}')
                    del agent_templates[a_prefix]
                    del pop_sizes[a_prefix]
                    del pop_max_idxs[a_prefix]
                    del pop_idxs[a_prefix]
                    split_pops.append(a_prefix)
                    pop_names.remove(a_prefix)
                pop_max_idxs[a_prefix] = max(idx, pop_max_idxs.get(a_prefix, 0))
                pop_sizes[a_prefix] = pop_sizes.get(a_prefix, 0) + 1
                pop_idxs[a_prefix] = pop_idxs.get(a_prefix, []) + [idx]
        for p in pop_names:
            if pop_max_idxs[p] >= pop_sizes[p]:
                for idx in pop_idxs[p]:
                    agent_templates[f'{p}_{idx}'] = deepcopy(agent_templates[p])
                    nonpop_ag_names.append(f'{p}_{idx}')
                del agent_templates[p]
                del pop_sizes[p]
                del pop_max_idxs[p]
                del pop_idxs[p]
                pop_names.remove(p)
        for pref, tmpt in agent_templates.items():
            tmpt['n'] = pop_sizes.get(pref, 1)
            if tmpt['n'] > 1:
                del tmpt['controller']['name']
        print(self.out_dir, 'XXXX')
        return HD({
            "seed": self.rng.bit_generator.seed_seq.entropy,
            "out_dir": str(ic(self.out_dir)),
            "world": self.world.__class__.__name__,
            "world_params": self.world.json,
            "sb_factory": self.sb_factory.__class__.__name__,
            "sb_factory_params": self.sb_factory.json,
            "publication_params": self.publications.json,
            "agent_populations": pop_names,
            'agent_names': nonpop_ag_names,
            "agent_templates": agent_templates,
            "rewards": [rw.NAME for rw in self.rewards],
            "reward_params": {rw.NAME: rw.json for rw in self.rewards}
        }).simplify()

    def add_reward(self, rew: Reward):
        self.rewards.append(rew)

    def run(self, days, steps_per_day, state_file:str=None, prefix=''): 
        if not self.rewards:
            raise AttributeError("model has no agent rewards")
        if state_file: # loading state a nice thing for later
            pass
        path_ = Path(self.out_dir)
        path_.mkdir(parents=True, exist_ok=True)
        for _ in range(days):
            print(f'=== DAY {_} ===')
            asyncio.run(self.day(steps_per_day)) 
            self.night()
        for r in self.rewards:
            r.record.to_parquet(path_ / f'{prefix}{r.NAME}_record.parquet')
        for i, table in enumerate(self.publications.tables):
            table['tree'] = table['tree'].apply(lambda x: f"{x}")
            table.to_parquet(path_ / f'{prefix}publication_{i}_end.parquet')
        for agent in self.agents:
            for i, table in enumerate(agent.ac.memory.tables):
                table['tree'] = table['tree'].apply(lambda x: f"{x}")
                table.to_parquet(path_ / f'{prefix}{agent.name}_mem_{i}.parquet')
        pd.DataFrame({
            agent.name: agent.day_rewards for agent in self.agents
        }).to_parquet(path_ / f'{prefix}day_rewards.parquet')


    async def day(self, steps):
        print('Good morning!')
        for agent in self.agents:
            agent.morning_routine(steps)
        print('Oh, what a lovely day')
        for _ in range(steps):
            self.not_done = self.agent_name_set.copy()
            self._reward_dict = {}
            shuffled_agents = copy(self.agents)
            self.rng.shuffle(shuffled_agents)
            tasks = [asyncio.create_task(agent.day_step()) for agent in shuffled_agents]
            _ = await asyncio.wait(tasks)
            for a in self.agents:
                for i, tbl in enumerate(a.ac.memory.tables):
                    for j, row in tbl.iterrows():
                        if row['exists'] and ((ic(row)['tree'].size() != row['size']) or (row['tree'].depth() != row['depth'])):
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
        print('What a beautiful sunset!')
        for agent in self.agents:
            agent.evening_routine()

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
            sb_factories:dict=None, 
            network_classes:dict=None, 
            tree_factory_classes:dict=None,
            mutators:dict=None,
            gp_systems:dict=None,
            sb_statfuncs:dict=None,
            rewards:dict=None
        ):
        self.worlds = {**self.worlds, **(worlds or {})}
        self.sb_factories = {**self.sb_factories, **(sb_factories or {})}
        self.network_classes = {**self.network_classes, **(network_classes or {})}
        self.tree_factory_classes = {**self.tree_factory_classes, **(tree_factory_classes or {})}
        self.mutators = {**self.mutators, **(mutators or {})}
        self.gp_systems = {**self.gp_systems, **(gp_systems or {})}
        self.sb_statfuncs = {**self.sb_statfuncs, **(sb_statfuncs or {})}
        self.rewards = {**self.rewards, **(rewards or {})}
    
    def read_json(self, json_) -> HD:
        if not isinstance(json_, HD):
            if isinstance(json_, str):
                with open(json_) as f:
                    json_ = json.load(f)
            json_ = HD(json_)
        return json_

    def from_json(self, json_: str|dict) -> Model:
        json_ = self.read_json(json_)
        dancing_chaos_at_the_heart_of_the_world = np.random.Generator(
            np.random.PCG64(json_.get('seed', None))
        )
        world = self.worlds[json_['world']].from_json(json_)
        print(f'Seed: {dancing_chaos_at_the_heart_of_the_world.bit_generator.seed_seq.entropy}')
        sb_factory = self.sb_factories[json_['sb_factory']].from_json(json_)
        time = ModelTime()
        listed_names = json_.get('agent_names', [])
        agent_name_list = list(listed_names)
        for nom in json_[['agent_populations']]:
            if json_.get(["agent_templates", nom, "n"], 0) > 1:
                agent_name_list += [
                    f'{nom}_{i}' for i in range(
                        json_[["agent_templates", nom, "n"]]
                    )
                ]
            else:
                agent_name_list.append(nom)
        if len(set(agent_name_list)) < len(agent_name_list):
            for i, nm in enumerate(agent_name_list):
                if nm in agent_name_list[i+1:]:
                    raise ValueError(
                        f"Duplicate name: {nm} in {agent_name_list}"
                    )
        agent_names_2_idxs = {name: i for i, name in enumerate(agent_name_list)}
        pub = Publication.from_json(json_, time=time, agent_names=agent_names_2_idxs)
        agents = []
        for agdata in ["agent_names", "agent_populations"]:
            for nom in json_.get([agdata], []):
                if agdata=="agent_populations" and nom in listed_names:
                    raise ValueError(
                        f"{nom} cannot be used as both an agent name and a" +
                        "population prefix"
                    )
                tree_factory_classes = [
                    self.tree_factory_classes[tfc]
                    for tfc 
                    in json_[[
                        "agent_templates", 
                        nom, 
                        "controller", 
                        "tree_factory_classes"
                    ]]
                ]
                gp_system = self.gp_systems[
                    json_[[
                        "agent_templates", 
                        nom, 
                        "controller", 
                        "gp_system"
                    ]]
                ] 
                network_class = self.network_classes[
                    json_[[
                        "agent_templates", 
                        nom, 
                        "controller", 
                        "network_class"
                    ]]
                ]
                agents += [
                    Agent.from_json(
                        json_,
                        prefix=nom,
                        network_class=network_class,
                        rng=dancing_chaos_at_the_heart_of_the_world,
                        controller=AgentController.from_json(
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
                        )
                    ) for i in range(json_.get(['agent_templates', nom, 'n'], 1))
                ]
        # Note, this must be done after all agents have been made,
        # as some params depend on knowing how many other agents there are
        for agent in agents:
            prefix_ = agent.prefix
            agent.make_networks(
                **json_[['agent_templates', prefix_, 'network_params']]
            )
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
        for agent in agents:
            agent.ac.model = model
        ic.enable()
        for reward in json_['rewards']:
            model.add_reward(
                self.rewards[reward].from_json(json_, model=model)
            )
        return model

    def save_json_and_run(self,
            model: Model, 
            steps_per_day: int,
            days: int,
            prefix='philoso_py_',
            json_out_dir=None,
            run=True):
        saved_json = {
            **model.json,
            'days': days, 
            'steps_per_day': steps_per_day,
            'output_prefix': prefix
        }
        if json_out_dir is None:
            if "out_dir" in saved_json:
                json_out_dir = saved_json['out_dir']
            else:
                json_out_dir = ""
        print(json_out_dir)
        path = Path(json_out_dir)
        path.mkdir(parents=True, exist_ok=True)
        json_outpath = path / f'{prefix}params.json'
        with open(json_outpath, 'w', encoding='utf-8') as f:
            json.dump(
                saved_json, f, 
                ensure_ascii=False, 
                indent=2
            )
            print(json_outpath)
        if run:
            model.run(
                days, 
                steps_per_day,
                prefix=prefix
            )
    
    def run_json(self, json_: str|dict):
        json_ = self.read_json(json_)
        model = self.from_json(json_)
        days = json_['days']
        steps_per_day = json_['steps_per_day']
        output_prefix=json_['output_prefix']
        self.save_json_and_run(
            model, 
            steps_per_day,
            days,
            prefix=output_prefix)

if __name__ == "__main__":
    # model = example_model(seed=69420, ping_freq=10)
    # model.run(100,100, prefix='k__')
    # model.run(40, 100)
    # model.run(100, 100)
    # model.run(2, 10_000)
    # model.run(days, steps_per_day) # 
    # ModelFactory().run_json('model1.json')
    ModelFactory().save_json_and_run(
        example_model(
            seed=69, 
            out_dir='output/m', 
            ping_freq=5
        ),
        100,
        100,
        "m__"
    )