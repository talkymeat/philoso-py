"""based on PPO implementation by Edan Meyer (https://github.com/ejmejm)

Original file is located at
    https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb
"""

import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Discrete
from icecream import ic

# Since I'm using dictionaries to store outputs based on which
# action head they related to, I need functions to convert
# dicts of tensors to tensors

def logprobdict_2_tensor(logprob_dict):
    """Joins all the Tensors in an OrderedDict of Tensors into one Tensor"""
    for k, v in logprob_dict.items():
        if len(v.shape)==2 and v.shape[0]==1:
            logprob_dict[k] = v[0]
    return torch.concatenate(list(logprob_dict.values()))
    
def tensorise_dict(act_dict, device):
    """Converts a dict with string keys and list-like, array-like 
    or tensor-like values into an OrderedDict of Tensors
    """
    return OrderedDict({k: t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device) for k, t in act_dict.items()})


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
    def __init__(self, obs_space_size, action_space_sizes, device:str="cpu",):
        super().__init__() # manadatory
        # dict for action heads
        self.policy_layers: dict[str, nn.Sequential] = {}

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
        self.device = device
        # The following is an alternative scheme for actions options, in which 
        # `read_to_gp` adds trees directly to the treebank for a gp
        # self.action_choices = [
        #     [],
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
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64).double(),
            nn.ReLU().double(), 
            nn.Linear(64, 64).double(), 
            nn.ReLU().double()) 
        
        # choice head
        self.policy_layers['choice'] = nn.Sequential(
            nn.Linear(64, 64).double(), 
            nn.ReLU().double(), 
            nn.Linear(64, self.action_choice_space.n).double()) 
        
        # action heads
        for name, size in action_space_sizes.items():
            self.policy_layers[name] = nn.Sequential(
                nn.Linear(64, 64).double(),
                nn.ReLU().double(),
                nn.Linear(64, size).double())
        
        # value (critic) head
        self.value_layers = nn.Sequential(
            nn.Linear(64, 64).double(),  # DOUBLE
            nn.ReLU().double(), # DOUBLE
            nn.Linear(64, 1).double()) # DOUBLE

    def value(self, obs):
        z = self.shared_layers(obs)
        value = self.value_layers(z)
        return value
        
    def choose(self, obs):
        return self.policy(obs, 'choice')
        
    def policy(self, obs, choice):
        z = self.shared_layers(obs)
        action_logits = self.policy_layers[choice](z)
        return action_logits

    def forward(self, obs): # mandatory
        """Given an observation, will first call the choice head
        to select actions, when calls the actions heads corresponding
        to the chosen actions. Outputs the choice, the log probability
        of the choice, the action logits, and the predicted value
        """

        torch.autograd.set_detect_anomaly(True)

        # This didn't fix the inplace bug:
        # obs = obs.clone().detach().requires_grad_(True)

        # feeds the observation into the shared layers
        z = self.shared_layers(obs)
        action_logits = {}
        # output of the shared layer is used to get logits from the
        # 'choice' head
        action_logits['choice'] = self.policy_layers['choice'](z)
        # which is used to get a choice from a Categorical distribution
        choice_distribution = Categorical(logits=action_logits['choice'])
        choice = choice_distribution.sample()
        # the log prob is needed for training
        choice_log_prob = choice_distribution.log_prob(choice).item()
        # having made the choice, the output of the shared layer is also 
        # passed into the layers for each action, in sequence, generating
        # the action logits
        for action in self.action_choices[choice.item()]:
            action_logits[action] = self.policy_layers[action](z)
        # the critic layer gives the observation a value score
        value = self.value_layers(z)
        return choice, choice_log_prob, action_logits, value
  

class PPOTrainer():
    """Performs PPO training of ActorCriticNetwork"""
    def __init__(self,
                actor_critic: ActorCriticNetwork,
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
        self.policy_params = {}
        self.policy_optims = {}

        # all heads get Adam Optimsers
        value_params = list(self.actor_critic.shared_layers.parameters()) + \
            list(self.actor_critic.value_layers.parameters())
        self.value_optim = optim.Adam(value_params, lr=value_lr)

        for k, layers in self.actor_critic.policy_layers.items():
            self.policy_params[k] = list(
                self.actor_critic.shared_layers.parameters()
            ) + list(layers.parameters())
            self.policy_optims[k] = optim.Adam(self.policy_params[k], lr=policy_lr)

    @property
    def device(self):
        self.actor_critic.device

    def train_policy(self, obs, acts, old_log_probs, gaes, which, actions, mask=None):
        """Trains a single policy head, using the data from a given day (epoch)
        
        Parameters
        ----------
            obs: 
                Observation
            acts:
                The  actions actually taken during the day 
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
        for _ in range(self.max_policy_train_iters):
            # If a network is used at every step (like 'choice'), no mask needed
            obs, gaes = (obs[mask], gaes[mask]) if mask is not None else (obs, gaes)
            # acts = acts[mask] < Not needed, masking done in Agent.night
            # old_log_probs = old_log_probs[mask] < likewise

            self.policy_optims[which].zero_grad()

            # Here, we calculate the new log probs, which is different
            # (simpler) for 'choice' than everything else
            if which == 'choice':
                # a new choice based on the updated network params, and
                # corresponding log probs
                new_logits = self.actor_critic.choose(obs)
                new_distro = Categorical(logits=new_logits) 
                new_log_probs = new_distro.log_prob(acts)
            else:
                #
                new_logprobs_list = []
                for ob1, ac1 in zip(obs, acts):
                    # generate new logits
                    new_logits = self.actor_critic.policy(ob1, which)
                    # An `Action` may use different parts of its input in
                    # different ways, with different action subspaces
                    # and using different distributions. This generates
                    # a dict of the distributions, given the logits
                    new_distros = actions[which].logits_2_distros(new_logits)
                    # this goes through the new distributions and calculates
                    # the new log probabilities of the action-parts, based on
                    # the updated network parameters
                    act_part_log_probs = tensorise_dict(OrderedDict({
                        k: d.log_prob(ac1[k]) for k, d in new_distros.items()
                    }), device=self.device)
                    # the logprobs for the sub-actions are concatenated into
                    # a single tensor
                    new_logprobs_list.append(logprobdict_2_tensor(act_part_log_probs))
                # the log-probs for each action are then stacked into a 2-d tensor
                new_log_probs = torch.stack(
                    new_logprobs_list, 
                    dim=0
                ).to(self.device, dtype=torch.float64) 
                # Would this work? test later XXX XXX
                # new_logits = self.actor_critic.policy(obs, which)
                # new_distros = actions[which].logits_2_distros(new_logits)
                # But for now, do what I know will work :|

            # So at this point the old and new logprobs are both
            # 2-d arrays of matching size, assigned to the 
            # new_act = self.actor_critic.policy(obs)

            # the policy ratio is calculted and clipped to stop
            # the policy from changing too much in one epoch
            policy_ratio = torch.exp(new_log_probs - old_log_probs) # torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(
                1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            
            # there are multiple policy ratios per action, the
            # gaes tensor needs to be reshaped to multiply correctely
            if len(clipped_ratio.shape) > 1:
                gaes = gaes.expand(1, gaes.shape[0]).permute((1,0))

            # calculate policy lost
            clipped_loss = clipped_ratio * gaes
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()

            # backpropagate
            policy_loss.backward()
            self.policy_optims[which].step()

            # If the new probability distro diverges too far from the old,
            # stop training and get new training data
            kl_div = (old_log_probs - new_log_probs).mean() 
            if kl_div >= self.target_kl_div:
                break

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

