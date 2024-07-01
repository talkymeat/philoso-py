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
    for k, v in logprob_dict.items():
        if len(v.shape)==2 and v.shape[0]==1:
            logprob_dict[k] = v[0]
    try:
        return torch.concatenate(list(logprob_dict.values()))
    except Exception as e:
        print('69'*69)
        print(logprob_dict)
        print({k: v.shape for k, v in logprob_dict.items()})
        print('96'*69)
        raise e

def tensorise_dict(act_dict, device):
    return OrderedDict({k: t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device) for k, t in act_dict.items()})


# Policy and value model
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_sizes, device:str="cpu",):
        super().__init__() # manadatory
        self.policy_layers: dict[str, nn.Sequential] = {}

        torch.autograd.set_detect_anomaly(True)

        # self.last_policy_networks = [] # ??? XXX
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

        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 64).double(),
            nn.ReLU().double(),
            nn.Linear(64, 64).double(),
            nn.ReLU().double())
        
        self.policy_layers['choice'] = nn.Sequential(
            nn.Linear(64, 64).double(),
            nn.ReLU().double(),
            nn.Linear(64, self.action_choice_space.n).double())
        
        for name, size in action_space_sizes.items():
            self.policy_layers[name] = nn.Sequential(
                nn.Linear(64, 64).double(),
                nn.ReLU().double(),
                nn.Linear(64, size).double())
        
        """# Also, 'gp_new'. XXX DELETE commented out code once AC is known to work
        # self.policy_networks['gp_continue'] = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))
        
        # self.policy_networks['store_mem'] = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))
        
        # self.policy_networks['use_mem'] = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))
        
        # self.policy_networks['use_public'] = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))
        
        # self.policy_networks['publish'] = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))
        
        # self.policy_networks['read'] = nn.Sequential(
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, action_space_size))"""
        
        self.value_layers = nn.Sequential(
            nn.Linear(64, 64).double(),
            nn.ReLU().double(),
            nn.Linear(64, 1).double())

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

        torch.autograd.set_detect_anomaly(True)

        # obs = obs.clone().detach().requires_grad_(True)
        z = self.shared_layers(obs)
        # print(z)
        action_logits = {}
        action_logits['choice'] = self.policy_layers['choice'](z)
        choice_distribution = Categorical(logits=action_logits['choice'])
        choice = choice_distribution.sample()
        choice_log_prob = choice_distribution.log_prob(choice).item()
        for action in self.action_choices[choice.item()]:
            action_logits[action] = self.policy_layers[action](z)
        value = self.value_layers(z)
        return choice, choice_log_prob, action_logits, value
  

class PPOTrainer():
    def __init__(self,
                actor_critic: ActorCriticNetwork,
                ppo_clip_val=0.2,
                target_kl_div=0.01,
                max_policy_train_iters=80,
                value_train_iters=80,
                policy_lr=3e-4,
                value_lr=1e-2):
        self.actor_critic = actor_critic
        self.ppo_clip_val = ppo_clip_val
        self.target_kl_div = target_kl_div
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        self.policy_params = {}
        self.policy_optims = {}

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
        for _ in range(self.max_policy_train_iters):
            # If a network is used at every step (like 'choice'), no mask needed
            obs, gaes = (obs[mask], gaes[mask]) if mask is not None else (obs, gaes)
            # acts = acts[mask] < Not needed, masking done in Agent.night
            # old_log_probs = old_log_probs[mask] < likewise

            self.policy_optims[which].zero_grad()

            # Here, we calculate the new log probs, which is different
            # (simpler) for 'choice' than everything else
            if which == 'choice':
                new_logits = self.actor_critic.choose(obs)
                new_distro = Categorical(logits=new_logits) 
                new_log_probs = new_distro.log_prob(acts)
            else:
                new_logprobs_list = []
                for ob1, ac1 in zip(obs, acts):
                    new_logits = self.actor_critic.policy(ob1, which)
                    new_distros = actions[which].logits_2_distros(new_logits)
                    act_part_log_probs = tensorise_dict(OrderedDict({
                        k: d.log_prob(ac1[k]) for k, d in new_distros.items()
                    }), device=self.device)
                    new_logprobs_list.append(logprobdict_2_tensor(act_part_log_probs))
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

            policy_ratio = torch.exp(new_log_probs - old_log_probs) # torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(
                1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
            
            try:
                if len(clipped_ratio.shape) > 1:
                    gaes = gaes.expand(1, gaes.shape[0]).permute((1,0))
                clipped_loss = clipped_ratio * gaes
            except Exception as e:
                print("Xx"*200)
                print(f"{clipped_ratio=}")
                print(f"{which=}")
                print(f"{gaes=}")
                print("xX"*200)
                raise e
            full_loss = policy_ratio * gaes
            policy_loss = -torch.min(full_loss, clipped_loss).mean()

            policy_loss.backward()
            self.policy_optims[which].step()

            # If the new probability distro diverges too far from the old,
            # stop training and get new training data
            kl_div = (old_log_probs - new_log_probs).mean() # (old_log_probs - new_log_probs).mean()
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

