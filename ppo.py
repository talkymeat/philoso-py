"""based on PPO implementation by Edan Meyer (https://github.com/ejmejm)

Original file is located at
    https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb
"""
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Discrete
#from agent_controller import AgentController
# from world import SineWorld
# from gp import GPTreebank
# from tree_factories import TreeFactoryFactory, RandomPolynomialFactory
# import gymnasium as gym
# from gymnasium.spaces.utils import flatten, unflatten
from icecream import ic

def logprobdict_2_tensor(logprob_dict):
    return torch.concatenate(list(logprob_dict.values()))

def tensorise_dict(act_dict, device):
    return OrderedDict({k: torch.tensor(t, device=device) for k, t in act_dict})

#sns.set()

#DEVICE = "cpu"

# Policy and value model
class ActorCriticNetwork(nn.Module):
    def __init__(self, obs_space_size, action_space_sizes, device:str="cpu",):
        super().__init__() # manadatory
        self.policy_layers: dict[str, nn.Sequential] = {}
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
            nn.Linear(obs_space_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU())
        
        self.policy_layers['choice'] = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_choice_space.n))
        
        for name, size in action_space_sizes.items():
            self.policy_layers[name] = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, size))
        
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
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1))

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
        z = self.shared_layers(obs)
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


    def train_policy(self, obs, acts, old_log_probs, gaes, which, actions, mask=None):
        for _ in range(self.max_policy_train_iters):
            # If a network is used at every step (like 'choice'), no mask needed
            obs = obs[mask] if mask else obs
            # acts = acts[mask] < Not needed, masking done in Agent.night
            # old_log_probs = old_log_probs[mask] < likewise
            gaes = gaes[mask] if mask else gaes

            self.policy_optims[which].zero_grad()

            # Here, we calculate the new log probs, which is different
            # (simpler) for 'choice' than everything else
            if which == 'choice':
                new_logits = self.actor_critic.choice(obs)
                new_distro = Categorical(logits=new_logits) 
                new_log_probs = new_distro.log_prob(acts)
            else:
                new_logprobs_list = []
                for ob1, ac1 in zip(obs, acts):
                    new_logits = self.actor_critic.policy(ob1, which)
                    new_distros = actions[which].logits_2_distros(new_logits)
                    act_part_log_probs = tensorise_dict(OrderedDict({
                        k: d.log_prob(ac1[k]) for k, d in new_distros.items()
                    }))
                    new_logprobs_list.append(act_part_log_probs)
                new_log_probs = torch.tensor(
                    new_logprobs_list,
                    dtype=torch.float32, 
                    device=self.device
                )
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
            
            clipped_loss = clipped_ratio * gaes
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


# def rollout(model, env, max_steps=1000):
#     """
#     Performs a single rollout.
#     Returns training data in the shape (n_steps, observation_shape)
#     and the cumulative reward.
#     """
#     ### Create data storage
#     train_data = pd.concat([
#         pd.DataFrame(
#             columns=pd.MultiIndex.from_product([
#                 ['obs', 'value', 'reward'], 
#                 ['v']
#             ])
#         ),
#         pd.DataFrame(
#             columns=pd.MultiIndex.from_product([
#                 ['choice', 'gp', 'store_mem', 'use_mem', 'publish', 'read'], 
#                 ['act', 'log_prob']
#             ])
#         )
#     ], axis=1)
#     obs, _ = env.reset() # XXX not sure I want this here
#     # XXX but I do need to know 

#     ep_reward = 0
#     for _ in range(max_steps):
#         choice, choice_log_prob, action_logits, val = model(torch.tensor(
#             [obs], dtype=torch.float32, device=DEVICE
#         ))
#         training_instance = {('obs', 'v'): obs, ('choice', 'act'): choice, ('choice', 'log_prob'): choice_log_prob}
#         action = {}
#         for act, logits in action_logits:
#             act_distribution = Categorical(logits=logits)
#             action[act] = training_instance[(act, 'act')] = act_distribution.sample()
#             training_instance[(act, 'log_prob')] = act_distribution.log_prob(act).item()

        
#         # logits is shape (1, len(action_space)), we want (len(action_space), )
#         act_raw = np.array(logits[0].detach())
#         # ========
#         # act_log_prob = act_distribution.log_prob(act).item()
#         # XXX process act output for env.step(act), but for now:
#         act = act_raw

#         # act, val = act.item(), val.item()
#         # This is a tensor of size (1,1), we just want a float
#         val = val.item()
#         training_instance[('value', 'v')] = val

#         next_obs, reward, done, _, __, ___ = env.step(action)

#         training_instance[('reward', 'v')] = reward

#         train_data.loc[len(train_data)] = training_instance

#         obs = next_obs
#         ep_reward += reward
#         if done:
#             break

#     train_data = [np.asarray(x) for x in train_data]

#     ### Do train data filtering
#     train_data[3] = calculate_gaes(train_data[2], train_data[3])

#     return train_data, ep_reward



# if __name__ == "__main__":
#     # env = gym.make('CartPole-v0')\
#     env = AgentController(
#         SineWorld(10.0, 100, 0.01, (1., 1., 0.), (0.01, 0.01, 0.005)),
#         GPTreebank,
#         fitness_measures = ['imse', 'size', 'depth'],
#         tree_factory_factories=[TreeFactoryFactory(RandomPolynomialFactory)]
#     )

#     model = ActorCriticNetwork(env.observation_space.shape[0], env.action_space.shape[0])
#     model = model.to(DEVICE)




#     # Define training params
#     n_episodes = 30 # 'days'
#     print_freq = 20

#     ppo = PPOTrainer(
#         model,
#         policy_lr = 3e-4,
#         value_lr = 1e-3,
#         target_kl_div = 0.02,
#         max_policy_train_iters = 40,
#         value_train_iters = 40)





#     # Training loop <<==== NIGHT
#     ep_rewards = []
#     for episode_idx in range(n_episodes):
#         # Perform rollout
#         train_data, reward = rollout(model, env)
#         ep_rewards.append(reward)

#         # Shuffle
#         permute_idxs = np.random.permutation(len(train_data[0]))

#         # Policy data
#         obs = torch.tensor(train_data[0][permute_idxs],
#                             dtype=torch.float32, device=DEVICE)
#         acts = torch.tensor(train_data[1][permute_idxs],
#                             dtype=torch.int32, device=DEVICE)
#         gaes = torch.tensor(train_data[3][permute_idxs],
#                             dtype=torch.float32, device=DEVICE)
#         act_2 = torch.tensor(train_data[4][permute_idxs],
#                                     dtype=torch.float32, device=DEVICE) # was log_probs

#         # Value data
#         returns = discount_rewards(train_data[2])[permute_idxs]
#         returns = torch.tensor(returns, dtype=torch.float32, device=DEVICE)

#         # Train model
#         ppo.train_policy(obs, acts, act_2, gaes) # act2 was log_probs
#         ppo.train_value(obs, returns)

#         if (episode_idx + 1) % print_freq == 0:
#             print('Episode {} | Avg Reward {:.1f}'.format(
#                 episode_idx + 1, np.mean(ep_rewards[-print_freq:])))