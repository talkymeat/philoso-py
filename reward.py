from abc import ABC, abstractmethod
from m import MDict


import numpy as np

class Reward(ABC):
    def __init__(self, 
            model,
            *args
        ) -> None:
        self.model = model

    def __call__(self) -> MDict[str, float]:
        return self.process_reward(*self.get_reward_data())

    @abstractmethod
    def process_reward(self, *args) -> MDict[str, float]:
        pass

    @abstractmethod
    def get_reward_data(self):
        pass


class Curiosity(Reward):
    def __init__(self, 
            model,
            def_fitness: str,
            first_finding_bonus: float,
            *args
        ) -> None:
        super().__init__(model)
        #self.agent_names: dict[str, int] = self.model.agent_names
        self.agents: dict[str, 'Agent'] = {a.name: a for a in self.model.agents}
        self.mean_mem_fitness_dict = {k: 0.0 for k in self.agents.keys()}
        self.def_fitness = def_fitness
        self.first_finding_bonus = first_finding_bonus

    def process_reward(self) -> MDict[str, float]:
        return MDict(self.get_reward_data())
    
    def get_reward_data(self):
        return {a.name: self.agent_reward_data(a) for a in self.agents}

    def agent_reward_data(self, agent):
        mean_tree_fitness = np.array(
            [agent.ac.memory[self.def_fitness] for agent in self.agents]
        ).mean()
        if mean_tree_fitness > 0 and self.mean_mem_fitness_dict[agent.name] == 0:
            self.mean_mem_fitness_dict[agent.name] = mean_tree_fitness
            return self.first_finding_bonus
        return mean_tree_fitness/self.mean_mem_fitness_dict[agent.name] - 1.0

            
class Renoun(Reward):
    def __init__(self, 
            model,
            *args
        ) -> None:
        super().__init__(model)

    def __call__(self) -> MDict[str, float]:
        return self.get_reward_data(*self.get_reward_data())

    def process_reward(self, rewards) -> MDict[str, float]:
        return MDict({row['agent']: row['reward'] for row in rewards.iloc})

    def get_reward_data(self):
        return self.model.publications.agents