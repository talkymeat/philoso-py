from abc import ABC, abstractmethod
from m import MDict
from icecream import ic
from utils import _i
from jsonable import SimpleJSONable
from hd import HierarchicalDict as HD

import numpy as np
import pandas as pd


DEBUG = True

def _print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

class Reward(SimpleJSONable, ABC):
    @property
    @abstractmethod
    def __name__(cls):
        raise NotImplementedError

    def __init__(self, 
            model,
            *args
        ) -> None:
        self.model = model
        self.record = pd.DataFrame({
            f'{ag.name}__{self.__name__}': []
            for ag in self.model.agents
        })

    def __call__(self) -> MDict[str, float]:
        return self.process_reward(self.get_reward_data())

    def process_reward(self, rewards, *args) -> MDict[str, float]:
        _print(f'{self.__name__} rewards:')
        _print(rewards)
        self.record.loc[len(self.record)] = {
            f'{k}__{self.__name__}': v
            for k, v in rewards.items()
        }
        return MDict(rewards)

    @abstractmethod
    def get_reward_data(self):
        pass

    @property
    def json(self)->dict:
        return {
            "name": self.__name__
        }

    @classmethod
    def from_json(cls, json_, *args, model=None, **kwargs):
        json_ = HD(json_)
        cls.addr = ['reward_params', cls.__name__]
        return cls(
            model,
            *[
                json_.get(cls.addr+[arg], None) 
                for arg 
                in cls.args
            ],
            *(json_.get(cls.addr+[cls.stargs], ()) if cls.stargs else ()),
            **{
                kwarg: json_[cls.addr+[kwarg]] 
                for kwarg 
                in cls.kwargs 
                if cls.addr+[kwarg] in json_
            }
        )


class Curiosity(Reward):
    """Gathers rewards for all agents besed on the quality (fitness) of the trees
    in each agent's own memory repositories.
    
    >>> from philoso_py import example_model
    OK
    >>> from gp import GPTreebank
    >>> from test_materials import T0, T1, T2, T3, T4, T5
    >>> from utils import aeq
    >>> model = example_model(seed=42)
    Seed: 42
    >>> model.agents[0].ac.gptb_list[0] = GPTreebank()
    >>> tt0 = T0.copy_out(model.agents[0].ac.gptb_list[0])
    >>> dummy_data = {'mse': 1.0, 'rmse': 1.0, 'wt_depth': 0.2, 'crossover_rate': 0.2, 
    ...     'wt_size': 0.3, 'obs_stop':10.1, 'depth': 4, 'pop': 99, 'mutation_sd': 0.4, 'max_depth': 25, 
    ...     'mutation_rate': 0.3, 'wt_fitness': 1.0, 'max_size': 55, 'raw_fitness': 0.9, 
    ...     'temp_coeff': 0.5, 'size': 13, 'obs_start':9.9, 'obs_num': 100, 'elitism':0.1}
    >>> model.agents[0].ac.memory.insert_tree(tt0, 0, 0, fitness=0.9, **dummy_data)
    >>> aeq(model.rewards[0]()['a0'], 1.0)
    True
    >>> model.t.tick()
    >>> aeq(model.rewards[0]()['a0'], 0.0)
    True
    >>> model.t.tick()
    >>> tt1 = T1.copy_out(model.agents[0].ac.gptb_list[0])
    >>> model.agents[0].ac.memory.insert_tree(tt1, 1, 0, fitness=2.7, **dummy_data)
    >>> aeq(model.rewards[0]()['a0'], 3.0)
    True
    >>> model.t.tick()
    >>> tt2 = T2.copy_out(model.agents[0].ac.gptb_list[0])
    >>> model.agents[0].ac.memory.insert_tree(tt2, 1, 0, fitness=0.9, **dummy_data)
    >>> aeq(model.rewards[0]()['a0'], -0.5)
    True
    >>> model.t.tick()
    >>> aeq(model.rewards[0]()['a0'], -0.5)
    True
    >>> model.t.tick()
    >>> tt3 = T3.copy_out(model.agents[0].ac.gptb_list[0])
    >>> model.agents[0].ac.memory.insert_tree(tt3, 1, 0, fitness=1.8, **dummy_data)
    >>> aeq(model.rewards[0]()['a0'], -0.25)
    True
    >>> model.t.tick()
    >>> tt4 = T4.copy_out(model.agents[0].ac.gptb_list[0])
    >>> model.agents[0].ac.memory.insert_tree(tt4, 1, 1, fitness=1.8, **dummy_data)
    >>> aeq(model.rewards[0]()['a0'], 0.25)
    True
    """
    __name__ = 'Curiosity'
    args = ["def_fitness", "first_finding_bonus"]

    def __init__(self, 
            model,
            def_fitness: str,
            first_finding_bonus: float,
            *args
        ) -> None:
        super().__init__(model)
        #self.agent_names: dict[str, int] = self.model.agent_names
        self.agents: dict[str, 'Agent'] = {a.name: a for a in self.model.agents}
        self.best_mean_fitness_dict = {k: 0.0 for k in self.agents.keys()}
        self.def_fitness = def_fitness
        self.first_finding_bonus = first_finding_bonus
    
    def get_reward_data(self):
        return {nm: self.agent_reward_data(a) for nm, a in self.agents.items()}

    def agent_reward_data(self, agent):
        # if len(agent.ac.memory)==0:
        #     return 0.0
        mean_tree_fitness = agent.ac.memory[self.def_fitness].mean()
        if self.best_mean_fitness_dict[agent.name] == 0:
            if mean_tree_fitness > 0:
                self.best_mean_fitness_dict[agent.name] = mean_tree_fitness
                return self.first_finding_bonus
            else: 
                return 0.0
        rew = mean_tree_fitness/self.best_mean_fitness_dict[agent.name] - 1.0
        self.best_mean_fitness_dict[agent.name] = max(
            mean_tree_fitness, 
            self.best_mean_fitness_dict[agent.name]
        )
        return rew
    
    @property
    def json(self):
        return {
            'def_fitness': self.def_fitness,
            'first_finding_bonus': self.first_finding_bonus,
            **super().json
        }
            
class Renoun(Reward):
    """Gathers rewards for all agents besed on their contributin to shared repositories
    
    >>> from philoso_py import example_model
    OK
    >>> from gp import GPTreebank
    >>> from test_materials import T0, T1, T2, T3, T4, T5
    >>> from utils import aeq
    >>> model = example_model(seed=42)
    Seed: 42
    >>> model.agents[0].ac.gptb_list[0] = GPTreebank()
    >>> tt0 = T0.copy_out(model.agents[0].ac.gptb_list[0])
    >>> dummy_data = {'mse': 1.0, 'rmse': 1.0, 'wt_depth': 0.2, 'crossover_rate': 0.2, 
    ...     'wt_size': 0.3, 'obs_stop':10.1, 'depth': 4, 'pop': 99, 'mutation_sd': 0.4, 'max_depth': 25, 
    ...     'mutation_rate': 0.3, 'wt_fitness': 1.0, 'max_size': 55, 'raw_fitness': 0.9, 
    ...     'temp_coeff': 0.5, 'size': 13, 'obs_start':9.9, 'obs_num': 100, 'elitism':0.1}
    """
    __name__ = 'Renoun'

    def __init__(self, 
            model,
            *args
        ) -> None:
        super().__init__(model)

    def get_reward_data(self):
        return self.model.publications.rewards()


class GuardrailCollisions(Reward):
    __name__ = 'GuardrailCollisions'

    def __init__(self, 
            model,
            *args
        ) -> None:
        super().__init__(model)
        #self.agent_names: dict[str, int] = self.model.agent_names
        self.agent_gms: dict[str, 'GuardrailManager'] = {a.name: a.ac.guardrail_manager for a in self.model.agents}
    
    def get_reward_data(self):
        return {nm: _i(gm.reward) for nm, gm in self.agent_gms.items()}


class Punches(Reward):
    __name__ = 'Punches'

    def __init__(self, 
            model,
            *args
        ) -> None:
        super().__init__(model)
        #self.agent_names: dict[str, int] = self.model.agent_names
    
    def get_reward_data(self):
        return {ag.name: _i(ag.ac.tmp.get('punches', 0)) for ag in self.model.agents}

def main():
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    main()