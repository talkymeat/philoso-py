from agent_controller import AgentController
from ppo import PPOTrainer, ActorCriticNetwork, calculate_gaes, discount_rewards, logprobdict_2_tensor, tensorise_dict
import pandas as pd
import numpy as np
import torch
from gymnasium.spaces import flatten, flatten_space
from action import Action
from collections import OrderedDict
from icecream import ic
from typing import Sequence
import pyarrow.parquet as pq
import pyarrow as pa

def filter_and_stack(ser: pd.Series, permute: Sequence[int], mask: Sequence[bool]):
    filtered = ser[permute][mask]
    acts: torch.Tensor = torch.stack(tuple(filtered))
    return acts

class Agent:
    def __init__(self, 
        ac:AgentController, # remember, ac is a gymnasium.env
        rng:np.random.Generator,
        device:str="cpu",
    ):
        self.rng=rng
        self.device = device
        self.done = False
        self.day_rewards = []
        # This df gets wiped at the start of each rollout
        self.ac = ac

    def make_networks(self,
                ppo_clip_val=0.2,
                target_kl_div=0.01,
                max_policy_train_iters=80,
                value_train_iters=80,
                policy_lr=3e-4,
                value_lr=1e-2
            ):
        self.ac.make_actions()
        self.actions: dict[Action] = self.ac.actions
        self.ac.make_observations()
        self.nn = ActorCriticNetwork( # put a factory class here, from param
            flatten_space(self.ac.observation_space).shape[0],
            self.ac.actions
            # {k: flatten_space(sp).shape[0] for k, sp in self.ac.action_space.items()}
        )
        self.nn.to(self.device)
        # Set up the training buffer with a multi-index
        self.policy_names = [('choice')]
        for k, head in self.nn.policy_heads.items():
            self.policy_names += [(k, kk) for kk in head.keys()]
        # +list(self.nn.policy_layers.keys())
        self.training_buffer_keys = [
            'obs', 'value', 'reward', ('act', 'choice'), ('log_prob', 'choice')
        ] + [
            (a_lp, *polname)
            for a_lp 
            in ['act', 'log_prob']
            for polname
            in self.policy_names
        ]
        # self.training_buffer = pd.concat([
        #     pd.DataFrame(
        #         columns=pd.MultiIndex.from_product([
        #             ['obs', 'value', 'reward'], 
        #             ['v']
        #         ])
        #     ),
        #     pd.DataFrame(
        #         columns=pd.MultiIndex.from_product([
        #             ['act', 'log_prob'],
        #             self.policy_names
        #         ])
        #     )
        # ], axis=1)
        self.trainer = PPOTrainer(
            self.nn,
            ppo_clip_val=ppo_clip_val,
            target_kl_div=target_kl_div,
            max_policy_train_iters=max_policy_train_iters,
            value_train_iters=value_train_iters,
            policy_lr=policy_lr,
            value_lr=value_lr
        )

    def __str__(self):
        return self.name

    @property
    def name(self):
        return self.ac.name

    # Set-up for rollout
    def morning_routine(self, steps):
        self.obs, _ = self.ac.reset(self.rng.bit_generator.seed_seq.entropy)
        self.training_buffer = pd.DataFrame({
            k: [None]*steps for k in self.training_buffer_keys
        })
        self.day_reward = 0
        self.done = False
        self.steps_done = 0

    # one action: a day is a loop of these
    async def day_step(self):
        # NN outputs: action_logits is raw NN output, a 1D tensor of floats.
        # choice is a simpler task as it's just a single Categorical, not a Dict
        obs = torch.tensor(
            [self.obs], dtype=torch.float64, device=self.device
        )
        try:
            choice, choice_log_prob, action_logits, val = self.nn(obs)
        except Exception as e:
            # remove after debugging - prints & outputs diagnostic info
            from gymnasium.spaces import unflatten
            print(('A ' if obs[0].isnan().any() else 'B ')*100)
            print(list(obs[0]))
            print('-'*80)
            print(self.ac._observation_space)
            print('-'*80)
            print(unflatten(self.ac._observation_space, obs[0].numpy()))
            print('='*80)
            randnum = self.rng.integers(1000000)
            for i, gp in enumerate(self.ac.gptb_list):
                if gp is not None:
                    print('='*80)
                    print(gp.scoreboard)
                    shx, shy = gp.scoreboard.shape
                    fname = f'scoreboard_dump_{randnum}_{i}_{shx}x{shy}.parquet'
                    print(f'Dumping scoreboard {i} as {fname}, {shx} by {shy}')
                    gp.scoreboard['tree'] = gp.scoreboard['tree'].apply(lambda x: f"{x}")
                    gp.scoreboard.to_parquet(fname)
                    print('='*80)
            print('='*80)
            print('Z '*100)
            raise e
        # set the observaton, plus the act and obs for `choice`
        training_instance = {
            ('obs'): self.obs, 
            ('act', 'choice'): choice, 
            ('log_prob', 'choice'): choice_log_prob
        }
        # separate action-representations for doing actions and training the 
        # NN. Tidying up so a single representation will do is a future task
        action_to_do = OrderedDict()
        for act, logits in action_logits.items(): 
            if act=='choice':
                # 'choice' is guaranteed to be simple - just
                # a single Discrete space, using a Categorical distro,
                # so we'll handle it separately elsewhere
                pass
            else:
                # Converts the logits to a dict of Distributions
                act_distros = self.actions[act].logits_2_distros(logits)
                # which makes a dict of samples from the distributions
                action_part = OrderedDict({
                    k: d.sample() for k, d in act_distros.items()
                })
                # which is then used to make a dict of log_probs
                act_part_log_probs = OrderedDict({
                    k: d.log_prob(action_part[k]) for k, d in act_distros.items()
                })

                action_to_do[act] = action_part
                for k, action_subpart in action_part.items():
                    training_instance[('act', act, k)] = action_subpart
                    training_instance[('log_prob', act, k)] = act_part_log_probs[k]


        # This is a tensor of size (1,1), we just want a float
        val = val.item()
        training_instance[('value')] = val

        next_obs, reward, done, _, _, _ = await self.ac.step(action_to_do)

        training_instance[('reward')] = reward
        # print('*'*10, training_instance)
        # print(self.training_buffer)
        ##### self.training_buffer.loc[len(self.training_buffer)] = training_instance
        # This (the 3 lines below) is janky and stupid and I hate it, but the 
        # line commented out above triggers torch to try to convert the tensors
        # to numpy arrays, which then errors because they have requires_grad=True
        for k, v in training_instance.items():
            try:
                self.training_buffer[k][self.steps_done] = v
            except Exception as e:
                print('g'*500)
                print(f'{i=}, {k=}')
                print(self.training_buffer)
                print(self.training_buffer_keys)
                print('f'*500)
        # print(self.training_buffer)

        self.obs = next_obs
        self.day_reward = self.day_reward + reward
        self.done = done
        self.steps_done += 1

    # post-processing of rollout
    def evening_routine(self):
        # train_data = [np.asarray(x) for x in self.training_buffer]

        # XXX TODO XXX XXX make sure these are the right columns
        ### Do train data filtering
        self.training_buffer['reward'] = calculate_gaes(self.training_buffer['value'], self.training_buffer[('reward')])
        self.day_rewards.append(self.day_reward)

    # run training loop
    def night(self, parquet_fname=None):
        # if parquet_fname:
        #     self.training_buffer = pd.read_parquet(parquet_fname)
        # else:
        #     table = pa.Table.from_pandas(self.training_buffer)
        #     pq.write_table(table, self.ac.out_dir / f'rollout_{self.name}_dump.parquet')
        permute_idxs = self.rng.permutation(len(self.training_buffer))

        # Policy data
        acts = {}
        act_log_probs = {}
        masks = {}
        obs = torch.tensor(
            self.training_buffer[('obs')][permute_idxs],
            dtype=torch.float64, 
            device=self.device
        )
        gaes = torch.tensor(
            self.training_buffer[('value')][permute_idxs],
            dtype=torch.float64, 
            device=self.device
        )
        for pol_name in self.policy_names:
            # Slightly different cases depending on whether the policy
            # is 'choice' or something else, as 'choice' is a simple
            # single Discrete space, using a Categorical distribution.
            # Others may have a mixture of space-types and distributions.
            if pol_name == 'choice':
                # Just convert acts and logprobs into nice 2-d tensors
                acts[pol_name] = torch.tensor(
                    list(self.training_buffer[('act', pol_name)][permute_idxs]),
                    dtype=torch.int64, 
                    device=self.device
                )
                act_log_probs[pol_name] = torch.tensor(
                    list(self.training_buffer[('log_prob', pol_name)][permute_idxs]),
                    dtype=torch.float64, 
                    device=self.device
                )
                # no mask needed, as 'choice' is used at every step
                masks[pol_name] = None
            else:
                # we can filter acts and logprobs here, but the mask must also be passed to
                # train_policy, because it's needed to filter obs, reward, etc, as these are
                # needs for each policy area, with a different mask on. Also, note each mask is
                # generated from an already permuted DF
                masks[pol_name] = ~(self.training_buffer[('log_prob', *pol_name)][permute_idxs].isna())
                # keeping acts as dicts makes sense, as the act must be used to make new log_probs,
                # which requires being split between multiple Distributions. These should just be 
                # assigned to device ready for training
                
                acts[pol_name] = filter_and_stack(
                    self.training_buffer[('act', *pol_name)], 
                    permute_idxs, 
                    masks[pol_name]
                ) # .to(self.device, dtype=torch.float64) XXX ???
                act_log_probs[pol_name] = filter_and_stack(
                    self.training_buffer[('log_prob', *pol_name)], 
                    permute_idxs, 
                    masks[pol_name]
                ) # .to(self.device, dtype=torch.float64) XXX ???
                # act_dicts: pd.Series = self.training_buffer[('act', *pol_name)][permute_idxs][masks[pol_name]]
                # act_dicts = act_dicts.apply(lambda x: tensorise_dict(x, device=self.device))
                # acts[pol_name] = act_dicts
                # # logprobs, on the other hand, are at the heart of the loss calculation, so
                # # these need to be turned into one big 1-d tensor, and then stacked into a 2-d
                # # tensor, which can be used for the actual training as a single computation
                # log_prob_dicts: pd.Series = self.training_buffer[('log_prob', *pol_name)][permute_idxs][masks[pol_name]]
                # # A pd.Series (or 1-col DF?) of tensors
                # log_probs_vecs = log_prob_dicts.apply(logprobdict_2_tensor).to_list()
                # # which is now a 2-d tensor, as promised
                # if log_probs_vecs:
                #     act_log_probs[pol_name] = torch.stack(
                #         log_probs_vecs, 
                #         dim=0
                #     ).to(self.device, dtype=torch.float64)
                

        # Value data
        returns = discount_rewards(self.training_buffer[('reward')])[permute_idxs]
        returns = torch.tensor(returns, dtype=torch.float64, device=self.device)

        # Train model
        for name in self.policy_names:
            if name in acts:
                self.trainer.train_policy(
                    obs, 
                    acts[name], 
                    act_log_probs[name], 
                    gaes, 
                    name, 
                    None if name == 'choice' else self.actions[name[0]], 
                    mask=masks[name]
                )
        self.trainer.train_value(obs, returns)
