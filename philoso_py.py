from repository import Publication
from world import World, SineWorld
from observatories import SineWorldObservatoryFactory
from tree_factories import RandomPolynomialFactory, RandomTreeFactory, RandomAlgebraicTreeFactory
from ppo import ActorCriticNetworkTanh, ActorCriticNetwork
from agent_controller import AgentController
from agent import Agent
from gp_fitness import SimpleGPScoreboardFactory
from model_time import ModelTime
from reward import Reward, Curiosity, Renoun, GuardrailCollisions, Punches
from tree_funcs import sum_all
from mutators import single_leaf_mutator_factory, single_xo_factory

from typing import Container
import asyncio
from pathlib import Path
from copy import copy

from icecream import ic

import numpy as np
import pandas as pd

ic.disable()

class Model:
    def __init__(self,
        world: World,
        dancing_chaos_at_the_heart_of_the_world: np.random.Generator,
        agents: Container[Agent],
        publications: Publication=None,
        sb_factory: SimpleGPScoreboardFactory=None,
        #rewards: list[Reward]=None,
        time: ModelTime=None,
        ping_freq=10
    ):
        self.world = world
        self.agents: Container[Agent] = agents
        for agent in agents:
            agent.model = self
        self.agent_name_set = set([agent.name for agent in self.agents])
        self.publications = publications
        self.rng = dancing_chaos_at_the_heart_of_the_world
        self.sb_factory = sb_factory
        self.rewards = []
        self.t = time
        

    def add_reward(self, rew: Reward):
        self.rewards.append(rew)

    def run(self, days, steps_per_day, state_file:str=None, prefix=''): 
        if not self.rewards:
            raise AttributeError("model has no agent rewards")
        if state_file: # loading state a nice thing for later
            pass
        for _ in range(days):
            print(f'=== DAY {_} ===')
            asyncio.run(self.day(steps_per_day)) 
            self.night()
        for r in self.rewards:
            r.record.to_parquet(f'{prefix}{r.NAME}_record.parquet')
        for i, table in enumerate(self.publications.tables):
            table['tree'] = table['tree'].apply(lambda x: f"{x}")
            table.to_parquet(f'{prefix}publication_{i}_end.parquet')
        for agent in self.agents:
            for i, table in enumerate(agent.ac.memory.tables):
                table['tree'] = table['tree'].apply(lambda x: f"{x}")
                table.to_parquet(f'{prefix}{agent.name}_mem_{i}.parquet')
        pd.DataFrame({
            agent.name: agent.day_rewards for agent in self.agents
        }).to_parquet(f'{prefix}day_rewards.parquet')


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

def model_from_json(json: str) -> Model:
    pass

def model_from_file(fname: str) -> Model:
    pass

def example_model(seed: int=None, out_dir: str|Path=Path('output', 'test'), ping_freq=5) -> Model:
    dancing_chaos_at_the_heart_of_the_world = np.random.Generator(np.random.PCG64(seed))
    world = SineWorld(
        np.pi*5, 100, 0.05, (1,100), (0.1, 10), 
        seed=dancing_chaos_at_the_heart_of_the_world
    )
    obs_factory = SineWorldObservatoryFactory(world)
    print(f'Seed: {dancing_chaos_at_the_heart_of_the_world.bit_generator.seed_seq.entropy}')
    sb_factory = SimpleGPScoreboardFactory(
        ['irmse', 'size', 'depth', 'penalty', 'hasnans', 'fitness'],
        'y'
    )
    n_agents = 8
    gp_vars_core = [
        'mse', 'rmse', 'size', 'depth', 'raw_fitness', 'fitness', 'value'
    ]
    gp_vars_more = [
        'wt_fitness', 'wt_size', 'wt_depth', "crossover_rate", "mutation_rate", 
        "mutation_sd", "max_depth", "max_size", "temp_coeff", "pop", "elitism", 
        'obs_start', 'obs_stop', 'obs_num'
    ]
    time = ModelTime()
    agent_names = {f'a{i}': i for i in range(n_agents)}
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
                f'a{i}', # name,
                6, # mem_rows,
                3, # mem_tables,
                world.dv, # dv,
                'irmse', # str, def_fitness
                sb_factory, # SimpleGPScoreboardFactory, # Needs to be more general XXX TODO
                obs_factory, # ObservatoryFactory
                [RandomAlgebraicTreeFactory], #tree_factory_classes, # tree_factory_classes: list[type[TreeFactory]],
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
        ) for i in range(n_agents)
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
    return model


if __name__ == "__main__":
    model = example_model(seed=69, ping_freq=10)
    model.run(50,100, prefix='h__')
    # model.run(40, 100)
    # model.run(100, 100)
    # model.run(2, 10_000)
    # model.run(days, steps_per_day) # 