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
        return self.process_reward(self.get_reward_data())

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

    def process_reward(self, data) -> MDict[str, float]:
        return MDict(data)
    
    def get_reward_data(self):
        try:
            return {nm: self.agent_reward_data(a) for nm, a in self.agents.items()}
        except Exception as e:
            print("WMWMWMWMWM"*10)
            print(self.agents)
            print(self.model.agents)
            print("MWMWMWMWMW"*10)
            raise e


    def agent_reward_data(self, agent):
        try:
            mean_tree_fitness = agent.ac.memory[self.def_fitness].mean()
        except Exception as e:
            print('WUB'*40)
            print(self.agents)
            print('WUB '*30)
            raise e
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

    def process_reward(self, rewards) -> MDict[str, float]:
        return MDict({row['agent']: row['reward'] for row in rewards.iloc})

    def get_reward_data(self):
        return self.model.publications._agents