import tensorflow as tf
import numpy as np
from tf_agents.trajectories import TimeStep, StepType
from tf_agents.agents.ppo import ppo_agent
from tf_agents.specs import BoundedArraySpec
from tf_agents.environments.py_environment import PyEnvironment
from world import World
from tree_factories import TreeFactory
from gp import GPTreebank
import operators as ops
from dataclasses import dataclass




class PhilosoPyAgent(PyEnvironment):
    """Note that `PhilosoPyAgent` is seen as an 'environment' by the
    `PhilosoPyRLAgent`, which is the RL subsystem of `PhilosoPyAgent`,
    in charge of higher-level decisions like when to make observations,
    run GP, publish results, etc.
    """

    @dataclass
    class GPMemory:
        observation_params: dict
        observations: list
        treebank: GPTreebank

    def __init__(
            self, 
            world: World,
            operators: list[ops.Operator],
            tree_factories: list[TreeFactory],
            name: str
        ):
        self.world = world
        self.name = name
        self.action_spec = [
            world.observation_params().replace(name='observation'),
            world.observation_params().replace(name='train_gp_static'),
            world.observation_params().replace(name='train_gp_live'),
            world.observation_params().replace(name='test_gp_predict'),
            BoundedArraySpec(name='publish_bet'), # don't implement until ready for multi-agent
            BoundedArraySpec(name='take_bet'), # don't implement until ready for multi-agent
            BoundedArraySpec(name='read_repo'), # don't implement until ready for multi-agent
            BoundedArraySpec(name='add_gp_result_to_knowledge'), 
            BoundedArraySpec(name='add_repo_result_to_knowledge'), # don't implement until ready for multi-agent
            BoundedArraySpec(name='prune_knowledge'),
            BoundedArraySpec(name='prune_records')
        ] #: tf_agents.typing.types.NestedTensorSpec,
        self.knowledge = GPTreebank(operators=operators)
        # structure this - maybe make a dataclass?
        self.records: list[PhilosoPyAgent.GPMemory] = []

    def time_step_spec(self):
        """Return time_step_spec."""
        pass

    def observation_spec(self):
        """Return observation_spec."""
        pass

    def action_spec(self):
        """Return action_spec."""
        return self.action_spec

    def _reset(self):
        """Return initial_time_step."""
        pass

    def _step(self, action):
        """Apply action and return new time_step."""
        pass

class PhilosoPyRLAgent(ppo_agent.PPOAgent):
    def __init__(
            self, 
            environment: PhilosoPyAgent
        ):
        first_step = environment.step()
        super(PhilosoPyRLAgent, self).__init__(
            time_step_spec = first_step, #: tf_agents.trajectories.TimeStep,
            action_spec = environment.action_spec()
            # optimizer = '???', #: Optional[types.Optimizer] = None,
            # actor_net = '???', #: Optional[tf_agents.networks.Network] = None,
            # value_net = '???' #: Optional[tf_agents.networks.Network] = None
        )

print('did not crash')