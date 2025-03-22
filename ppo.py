"""based on PPO implementation by Edan Meyer (https://github.com/ejmejm)

Original file is located at
    https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb
"""

import numpy as np

from copy import deepcopy
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Discrete, flatten_space, unflatten
from icecream import ic



# Policy and value model
class ActorCriticNetwork(nn.Module):
    """Multi headed actor-critic ntwork for philoso.py Agents: the 
    agents choose from multiple possible actions and each action also
    has a set of action parameters. When the agents acts, it first
    uses the 'choice' head to choose which actions to perform (a *choice*
    consists of a sequence of one or more *actions*) and then uses the
    head(s) corresponding to the chosen action(s) to set the parameters 
    for the action
    """

    NORMALISER = nn.ReLU

    def __init__(self, obs_space_size, actions, device:str="cpu", seed=None):
        super().__init__() # manadatory
        torch.manual_seed(seed)
        # dict for action heads
        self.policy_layers: OrderedDict[str, nn.Sequential] = OrderedDict()
        self.policy_heads: OrderedDict[tuple[str], nn.Linear] = OrderedDict()

        # needed to find this bug! :\
        torch.autograd.set_detect_anomaly(True)

        # these are the choices the 'choice' head can choose from:
        # each action name is a key in the action head dictionary
        self.action_choices = [
            ['gp_new'],
            ['gp_continue'],
            ['use_mem', 'gp_new'], # note, actions will be performed in list order, if more than one is specified
            ['use_mem', 'gp_continue'], 
            ['store_mem'],
            ['publish'],
            ['read']
        ]
        self.action_choice_space = Discrete(len(self.action_choices))
        # The following is an alternative scheme for actions options, in which 
        # `read_to_gp` adds trees directly to the treebank for a gp
        # self.action_choices = [
        #     ['gp'],
        #     ['gp_continue'],
        #     ['gp', 'use_mem'],
        #     ['gp_continue', 'use_mem'],
        #     ['gp', 'read_to_gp'],
        #     ['gp_continue', 'read'],
        #     ['gp', 'read_to_gp', 'use_mem'],
        #     ['gp_continue', 'read', 'use_mem'],
        #     ['store_mem'],
        #     ['publish'],
        #     ['read'] <<== optional, reads into mem, not gp run
        # ] 

        # chared with the value (critic) head, the choice head,
        # and all action heads
        self.make_layers(obs_space_size, actions)
        self.device = device

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device
        self.to(device = torch.device(device))

    def make_layers(self, obs_space_size, actions):
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64).double(),
            self.NORMALISER().double(), 
            nn.Linear(64, 64).double(), 
            self.NORMALISER().double()) 
        
        # choice head
        self.policy_layers['choice'] = nn.Sequential(
            nn.Linear(64, 64).double(), 
            self.NORMALISER().double(), 
            nn.Linear(64, self.action_choice_space.n).double()) 
        self.add_module('choice', self.policy_layers['choice'])

        # action heads
        for name, action in actions.items():
            self.policy_layers[name] = nn.Sequential(
                nn.Linear(64, 64).double(),
                self.NORMALISER().double()
            )
            self.add_module(name, self.policy_layers[name])
            head_dict = OrderedDict()
            for head_name, sub_space in action.action_space.items():
                size = flatten_space(sub_space).shape[0]
                head_dict[head_name] = nn.Linear(64, size).double()
                self.add_module(f'{name}____{head_name}', head_dict[head_name])
            self.policy_heads[name] = head_dict
        
        # value (critic) head
        self.value_layers = nn.Sequential(
            nn.Linear(64, 64).double(),  
            self.NORMALISER().double(),
            nn.Linear(64, 1).double()) 

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value
        
    def choose(self, obs):
        z = self.shared_layers(obs)
        choice_logits = self.policy_layers['choice'](z)
        return choice_logits
        
    def policy(self, obs, choice):
        z_0 = self.shared_layers(obs)
        z_1 = self.policy_layers[choice](z_0)
        action_logits = {k: head(z_1) for k, head in self.policy_heads.items()}
        return action_logits
    
    def sub_policy(self, obs, choice, head_name):
        z_0 = self.shared_layers(obs)
        z_1 = self.policy_layers[choice](z_0)
        action_logits = self.policy_heads[choice][head_name](z_1)
        return action_logits

    def forward(self, obs): # mandatory
        """Given an observation, will first call the choice head
        to select actions, when calls the actions heads corresponding
        to the chosen actions. Outputs the choice, the log probability
        of the choice, the action logits, and the predicted value
        """

        torch.autograd.set_detect_anomaly(True)

        # feeds the observation into the shared layers
        z_0 = self.shared_layers(obs)
        action_logits = {}
        # output of the shared layer is used to get logits from the
        # 'choice' head
        action_logits['choice'] = self.policy_layers['choice'](z_0)
        # which is used to get a choice from a Categorical distribution
        choice_distribution = Categorical(logits=action_logits['choice'])
        choice = choice_distribution.sample()
        # the log prob is needed for training
        choice_log_prob = choice_distribution.log_prob(choice).item()
        # having made the choice, the output of the shared layer is also 
        # passed into the layers for each action, in sequence, generating
        # the action logits
        for action in self.action_choices[choice.item()]:
            # action_logits[action] = self.policy_layers[action](z_0)
            z_1 = self.policy_layers[action](z_0)
            action_logits[action] = {
                k: head(z_1) 
                for k, head 
                in self.policy_heads[action].items()
            }
        # the critic layer gives the observation a value score
        value = self.value_layers(z_0)
        return choice, choice_log_prob, action_logits, value
    
    def save(self):
        return self.state_dict()

    def load(self):
        pass
  

# Policy and value model
class ActorCriticNetworkTanh(ActorCriticNetwork):
    """Multi headed actor-critic ntwork for philoso.py Agents: the 
    agents choose from multiple possible actions and each action also
    has a set of action parameters. When the agents acts, it first
    uses the 'choice' head to choose which actions to perform (a *choice*
    consists of a sequence of one or more *actions*) and then uses the
    head(s) corresponding to the chosen action(s) to set the parameters 
    for the action
    """
    NORMALISER = nn.Tanh

    # def make_layers(self, obs_space_size, actions):
    #     self.shared_layers = nn.Sequential(
    #         nn.Linear(obs_space_size, 64).double(),
    #         nn.Tanh().double(), 
    #         nn.Linear(64, 64).double(), 
    #         nn.Tanh().double()) 
        
    #     # choice head
    #     self.policy_layers['choice'] = nn.Sequential(
    #         nn.Linear(64, 64).double(), 
    #         nn.Tanh().double(), 
    #         nn.Linear(64, self.action_choice_space.n).double()) 
        
    #     # action heads
    #     for name, action in actions.items():
    #         self.policy_layers[name] = nn.Sequential(
    #             nn.Linear(64, 64).double(),
    #             nn.Tanh().double()
    #         )
    #         head_dict = {}
    #         for head_name, sub_space in action.action_space.items():
    #             size = flatten_space(sub_space).shape[0]
    #             head_dict[head_name] = nn.Linear(64, size).double()
    #         self.policy_heads[name] = head_dict
        
    #     # value (critic) head
    #     self.value_layers = nn.Sequential(
    #         nn.Linear(64, 64).double(),  
    #         nn.Tanh().double(),
    #         nn.Linear(64, 1).double()) 

class PPOTrainer():
    """Performs PPO training of ActorCriticNetwork"""
    def __init__(self,
                actor_critic: ActorCriticNetwork,
                np_random: np.random.Generator,
                ppo_clip_val=0.2,
                target_kl_div=0.01,
                max_policy_train_iters=80,
                value_train_iters=80,
                policy_lr=3e-4,
                value_lr=1e-2):
        self.actor_critic = actor_critic
        # training updates are clipped within a range: a key feature of PPO
        self.ppo_clip_val = ppo_clip_val
        # Trainer stops training once the target Kulback-Liebler
        # Divergence is reached, or training has gone on for the max
        # number of iterations
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        self.np_random = np_random

        # learning rates saved here for easy access when saving a 
        # JSON file to recreate a given model
        self.value_lr = value_lr
        self.policy_lr = policy_lr

        # all heads get Adam Optimsers
        value_params = list(self.actor_critic.shared_layers.parameters()) + \
            list(self.actor_critic.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)

        self.policy_params = list(self.actor_critic.shared_layers.parameters())
        for k, layers in self.actor_critic.policy_layers.items():
            self.policy_params += list(layers.parameters())
            if k in self.actor_critic.policy_heads:
                for kk, head in self.actor_critic.policy_heads[k].items():
                    self.policy_params += list(head.parameters())
            else:
                self.policy_params += list(layers.parameters())
        self.policy_optim = optim.Adam(self.policy_params, lr=policy_lr)

    @property
    def device(self):
        self.actor_critic.device

    def train_policy(self, obses, acts, old_log_probs, gaeses, pol_names, actions):
        """Trains a single policy head, using the data from a given day (epoch)
        
        Parameters
        ----------
            obs: 
                Observation
            acts:
                The actions actually taken during the day 
            old_log_probs:
                The log probabilities of the day's actions
            gaes:
                Generalised Advantage Estimation is used to represent value
            which (str):
                Indicates which head is to be trained
            actions:
                Dict containing `Action` objects: these translate network 
                outputs into actual behaviour, and contain useful 
                information about an action, like the corresponding 'action
                space', and which probability distributions are used
        """
        pol_names = list(acts.keys())
        for _ in range(self.max_policy_train_iters):
            print(f'Training round {_}:', pol_names)
            if not pol_names:
                break
            self.policy_optim.zero_grad()
            new_log_probs = {}
            # self.np_random.shuffle(pol_names)
            policy_loss = None
            for name in pol_names:
                # Here, we calculate the new log probs, which is different
                # (simpler) for 'choice' than everything else
                if name == 'choice':
                    # a new choice based on the updated network params, and
                    # corresponding log probs
                    new_logits = self.actor_critic.choose(obses[name].clone().detach()) # 1
                    new_distro = Categorical(logits=new_logits) 
                    # new_log_probs = new_distro.log_prob(acts)
                else:
                    # for the action parameters, we train one head at a time
                    new_logits = self.actor_critic.sub_policy(obses[name].clone().detach(), *name) # 1
                    new_distro = actions[name[0]].distributions[name[1]](new_logits)
                new_log_probs[name] = new_distro.log_prob(acts[name]) 
                # So at this point the old and new logprobs are both
                # 2-d arrays of matching size, assigned to the 
                # new_act = self.actor_critic.policy(obs)

                # the policy ratio is calculted and clipped to stop
                # the policy from changing too much in one epoch

                policy_ratio = torch.exp(new_log_probs[name] - old_log_probs[name].clone().detach()) #3
                clipped_ratio = policy_ratio.clamp(
                    1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            
                # there are multiple policy ratios per action, the
                # gaes tensor needs to be reshaped to multiply correctely
                if len(clipped_ratio.shape) > 1 and gaeses[name].dim()==1:
                    gaeses[name] = gaeses[name].expand(1, gaeses[name].shape[0]).permute((1,0))
                    
                # calculate policy lost
                gaes = gaeses[name].detach() # 2
                clipped_loss = clipped_ratio * gaes # 2
                full_loss = policy_ratio * gaes # 2
                if policy_loss is None:
                    policy_loss = -torch.min(full_loss, clipped_loss).mean()
                else:
                    policy_loss -= torch.min(full_loss, clipped_loss).mean()
            
            # backpropagate *once* for the overall loss
            policy_loss.backward()
            
            # and optimise
            self.policy_optim.step()
            new_pol_names = []
            for name in pol_names:
                # If the new probability distro diverges too far from the old,
                # stop training and get new training data
                kl_div = (new_log_probs[name] - old_log_probs[name]).mean() 
                if kl_div < self.target_kl_div:
                    new_pol_names.append(name)
            pol_names = new_pol_names

    def train_value(self, obs, returns):
        for _ in range(self.value_train_iters):
            self.value_optim.zero_grad()

            values = self.actor_critic.value(obs)
            value_loss = (returns - values) ** 2
            value_loss = value_loss.mean()

            value_loss.backward()
            self.value_optim.step()


def discount_rewards(rewards, gamma=0.99):
    """
    Return discounted rewards based on the given rewards and gamma param.
    """
    new_rewards = [float(rewards[len(rewards)-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return np.array(new_rewards[::-1])

def calculate_gaes(rewards, values, gamma=0.99, decay=0.97):
    """
    Return the General Advantage Estimates from the given rewards and values.
    Paper: https://arxiv.org/pdf/1506.02438.pdf
    """
    next_values = np.concatenate([values[1:], [0]])
    deltas = [rew + gamma * next_val - val for rew, val, next_val in zip(rewards, values, next_values)]

    gaes = [deltas[-1]]
    for i in reversed(range(len(deltas)-1)):
        gaes.append(deltas[i] + decay * gamma * gaes[-1])

    return np.array(gaes[::-1])

