
from abc import ABC, abstractmethod
from gymnasium.spaces import Space
import numpy as np

class Actionable(ABC):
    @property
    def action_param_space(self) -> Space:
        return self._act_param_space
    
    @property
    def action_param_names(self) -> list[str]:
        return self._act_param_names
    
    def act(self, params: np.ndarray|dict|list, *args, **kwargs):
        if params not in self.action_param_space:
            raise ValueError(f"Invalid action parameters: {params}")