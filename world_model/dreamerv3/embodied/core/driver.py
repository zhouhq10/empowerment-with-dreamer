import collections

import numpy as np
import math
import random

from .basics import convert
from copy import deepcopy

from collections import defaultdict

import tensorflow_probability as tfp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

from copy import deepcopy


def transform_single(x, proj, bias):
    x = x.reshape(1, -1)
    hk = np.dot(x, proj)
    hk = np.tanh(hk)
    hk += bias
    hk = 1 * (hk > 0.5) - 1 * (hk < -0.5)
    return tuple(hk.squeeze().tolist())


def _hash_key(x, proj=None, bias=None):
    # Obtained from C-BET
    # Expects X is be batched even if it is a single observation
    # Always returns a batched output
    if proj is None:
        return [tuple(obs.flatten().tolist()) for obs in x]

    return [transform_single(obs, proj, bias) for obs in x]


class Driver:
    # designed to manage interactions with an environment for reinforcement learning tasks.
    # The Driver class provides mechanisms to run episodes, collect data,
    # and optionally calculate intrinsic rewards for exploration using pseudocount-based methods
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(
        self,
        env,
        obs_intrinsic_reward="nan",
        hash_bits=128,
        intr_reward_coeff=0.001,
        latents_intrinsic_reward="nan",
        ignore_extr_reward=False,
        z_dim=32,
        h_dim=4096,
        n_envs=8,
        **kwargs,
    ):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.count_reset_prob = 0.001
        self.intr_reward_coeff = intr_reward_coeff
        self.obs_counts = dict()
        self.state_counts = dict()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.n_envs = n_envs
        self.factorized_state_counts = [dict() for _ in range(z_dim)]
        self.change_counts = dict()
        # # transition_counts[state_index][action][next_state_index] = count
        # self.state_index = dict()
        self.obs_intrinsic_reward = (
            obs_intrinsic_reward  # [pseudonovelty, novelty, nan, pseudocebt, cebt]
        )
        self.latents_intrinsic_reward = latents_intrinsic_reward
        self.ignore_extrinsic_reward = ignore_extr_reward

        proj_size = np.prod(env.obs_space["image"].shape)
        proj_dim = hash_bits
        self.state_proj = self.state_bias = self.change_proj = self.change_bias = None

        if "pseudo" in obs_intrinsic_reward:
            self.state_proj = np.random.normal(0, 1, (proj_dim, proj_size, 1))
            self.state_bias = np.random.uniform(-1, 1, (proj_dim, 1))
            self.change_proj = np.random.normal(0, 1, (proj_dim, proj_size, 1))
            self.change_bias = np.random.uniform(-1, 1, (proj_dim, 1))

        self.reset()

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value

    def update_counts(self, var, keys):
        for key in keys:
            if key in var:
                var[key] += 1
            else:
                var[key] = 1

    def comp_novelty_entropy(self, d):
        total = sum(d.values())
        if total == 0:
            # Edge case: no data
            return 0.0

        entropy = 0.0
        for count in d.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy

    def sample_action_sequences(self, horizon=2, num_samples=1e2):
        import ipdb

        ipdb.set_trace()

        def sample_one():
            import ipdb

            ipdb.set_trace()
            return {
                k: np.stack([v.sample() for _ in range(len(self._env))])
                for k, v in self._env.act_space.items()
                if k != "reset"
            }

        return [[sample_one() for _ in range(horizon)] for _ in range(num_samples)]

    def one_step_action_sequence(self, action, cur_obs, cur_state, policy):
        import ipdb

        ipdb.set_trace()
        # assert all(len(x) == len(env_copy) for x in action.values())
        # action = {k: v for k, v in action.items() if not k.startswith("log_")}
        # obs = env_copy.step(
        #     action
        # )  # obs is a dictionary; image size 1, 64, 64, 3 # BatchEnv.step
        # obs = {k: convert(v) for k, v in obs.items()}
        # assert all(len(x) == len(env_copy) for x in obs.values()), obs

        action, next_state, _ = policy(cur_obs, cur_state, **self._kwargs)
        action = {k: convert(v) for k, v in action.items()}
        return action, next_state

    def marginalize_action_entropy(self, conditioned_counts, action_probs=None):
        """
        Marginalize over actions to estimate H(S_{t+n} | S_t=s) from H(S_{t+n} | A_t, S_t=s).

        Args:
            conditioned_counts: dict of dicts {action: {next_state: count}}
            action_probs: dict {action: probability}, defaults to uniform distribution.

        Returns:
            float: Estimated entropy H(S_{t+n} | S_t=s)
        """
        actions = list(conditioned_counts.keys())
        if action_probs is None:
            # Assume uniform distribution over actions if not provided
            action_probs = {action: 1 / len(actions) for action in actions}

        # Aggregate next state counts across actions, weighted by action probabilities
        total_counts = defaultdict(float)
        for action, state_counts in conditioned_counts.items():
            for state, count in state_counts.items():
                total_counts[state] += action_probs[action] * count

        # Normalize to form a probability distribution
        total = sum(total_counts.values())
        prob_dist = {state: count / total for state, count in total_counts.items()}

        def compute_entropy(prob_dist):
            """Compute entropy from a probability distribution."""
            probs = np.array(list(prob_dist.values()))
            return -np.sum(probs * np.log2(probs + 1e-9))  # Avoid log(0)

        # Compute entropy H(S_{t+n} | S_t=s)
        return compute_entropy(prob_dist)

    def estimate_empowerment_sample(
        self, init_state, policy, init_obs, horizon=1, num_samples=1e2
    ):
        # TODO: longer horizon
        # TODO: non-random policy

        # Estimate H(Z' | a, z) by imagining the consequences of taking each action A=a in state Z=z (=init_state)
        H_Z_prime_given_z_and_a = []
        p_Z_prime_given_z_and_a = []
        n_actions = self._env.act_space["action"].shape[0]
        for action_idx in range(n_actions):  # TODO: hacky
            one_hot_action = np.zeros(n_actions, dtype=np.float32)
            one_hot_action[action_idx] = 1.0

            # Input new (hypothetical) action into policy
            cur_action = np.stack([one_hot_action for _ in range(len(self._env))])
            cur_state = deepcopy(init_state)
            cur_new_state = (
                (cur_state[0][0], cur_state[0][1], cur_action),
                cur_state[1],
                cur_state[2],
            )

            # Sample next state from policy
            cur_obs = deepcopy(init_obs)
            _, _, next_state = policy(cur_obs, cur_new_state, **self._kwargs)
            del cur_state, cur_obs, cur_new_state
            next_state_logit = next_state["logit"]
            next_state_prob = np.exp(next_state_logit) / np.sum(
                np.exp(next_state_logit), axis=-1, keepdims=True
            )
            next_state_entropy = -np.sum(
                next_state_prob * np.log(next_state_prob + 1e-12), axis=-1
            )

            H_Z_prime_given_z_and_a.append(next_state_entropy)
            p_Z_prime_given_z_and_a.append(next_state_prob)

        # Estimate the conditional entropy H(Z' | A, z) = sum_a p(a | z) H(Z' | a, z) 
        # where p(a | s) is the source policy (currently uniform)
        H_Z_prime_given_z_and_A = np.array(H_Z_prime_given_z_and_a).mean(0)

        # Estimate H(Z' | z) (by marginalizing over p(Z'|z,a) to obtain p(Z'|z))
        p_Z_prime_given_z = np.mean(np.array(p_Z_prime_given_z_and_a), axis=0)
        H_Z_prime_given_z = -np.sum(p_Z_prime_given_z * np.log(p_Z_prime_given_z + 1e-12), axis=-1)

        # Mutual Information: I(Z' ; A | s) = H(Z' | s) - H(S' | Z, s)
        # average over latent states
        average_MI = (H_Z_prime_given_z - H_Z_prime_given_z_and_A).mean(axis=-1)
        # ensure that MI is non-negative
        average_MI = np.maximum(average_MI, 0)
        return average_MI

    def reset(self):
        self._acts = {
            k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
            for k, v in self._env.act_space.items()
        }
        self._acts["reset"] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

        # Used for intrinsic reward
        self._current_obs_img = np.zeros(
            (len(self._env),) + self._env.obs_space["image"].shape,
            self._env.obs_space["image"].dtype,
        )

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        # Convert the actions to the correct data type
        assert all(len(x) == len(self._env) for x in self._acts.values())
        acts = {k: v for k, v in self._acts.items() if not k.startswith("log_")}
        obs = self._env.step(
            acts
        )  # obs is a dictionary; image size 1, 64, 64, 3 # BatchEnv.step
        obs = {k: convert(v) for k, v in obs.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs

        # Setting extrinsic reward
        extr_reward = obs["reward"]
        obs["extrinsic_reward"] = extr_reward
        obs["reward"] = (
            np.zeros_like(obs["reward"])
            if self.ignore_extrinsic_reward
            else obs["reward"]
        )

        # TODO: Seems like we should compute empowerment and novelty based on self._state here 
        # (when it is still z based on s, before it is updated to be z' based on s')
        previous_posterior = None
        if self.latents_intrinsic_reward != "nan":
            previous_posterior = deepcopy(self._state)

        # Calculate intrinsic reward on image observation
        if self.obs_intrinsic_reward != "nan":
            obs = self.calc_obs_intrinsic_reward(obs)
        self._current_obs_img = obs["image"]

        # Get the actions from current policy
        acts, self._state, current_prior = policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}

        # Use the latents from the world model for learned intrinsic reward
        if self.latents_intrinsic_reward != "nan":
            current_posterior = deepcopy(self._state)
            obs = self.calc_latent_intrinsic_reward(
                previous_posterior, current_prior, current_posterior, policy, obs
            )
            del previous_posterior
            del current_posterior

        # flags the end of an episode for each environment in a batch
        if obs["is_last"].any():
            mask = 1 - obs["is_last"]
            acts = {
                k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()
            }  # Sets the actions to zero for environments where the episode has ended, preventing unintended updates.

        # Update the reset flag
        acts["reset"] = obs["is_last"].copy()
        self._acts = acts

        # trns (short for "transitions") merges the observation (obs) and action (acts) dictionaries into a unified structure.
        trns = {**obs, **acts}

        # For environments where is_first is True, the corresponding episode buffer (self._eps[i]) is cleared to start fresh.
        if obs["is_first"].any():
            for i, first in enumerate(obs["is_first"]):
                if first:
                    self._eps[i].clear()

        # Update the replay buffer
        for i in range(len(self._env)):
            try:
                trn = {k: v[i] for k, v in trns.items()}

                # Save which environment the transition belongs to for debugging, dirty hack
                # TODO: this is a bit dirty, better to add to logger instead!
                trn['env_id'] = i  
            except:
                import ipdb

                ipdb.set_trace()
            # Append each observation and action to the corresponding episode buffer.
            [self._eps[i][k].append(v) for k, v in trn.items()]
            # Step increment and replay buffer addition
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1

        # For environments where is_last is True, the corresponding episode buffer (self._eps[i]) is copied and passed to the on_episode callbacks.
        if obs["is_last"].any():
            for i, done in enumerate(obs["is_last"]):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode
    
    def compute_surprise(self, counts_dict, z):
        # Calculate total count of all states observed so far
        total_visits = sum(counts_dict.values()) if counts_dict else 0
        state_count = counts_dict.get(z, 0)

        # Calculate novelty (surprise) after update 
        # (actual updating is only done later for all environments at once)
        surprise = -np.log((state_count + 1) / (total_visits + 1))
        
        # # Calculate normalization factor: maximum possible surprise = completely new state
        # # probability of new state after update: p=1/(N_states + 1)
        # total_states = len(counts_dict) if counts_dict else 1
        # max_surprise = np.log(total_states + 1)
        
        # # Normalized novelty
        # if max_surprise > 0:
        #     novelty_reward = surprise / max_surprise
        # else:
        #     # edge case: in first step, there is no other state to compare to, 
        #     # so log(1) = 0 and the surprise would be 0, and the max_surprise would be 0 as well
        #     # making the novelty reward undefined
        #     # We manually set it to 1.0, meaning that the first state is considered completely novel
        #     novelty_reward = 1.0    
        
        # return novelty_reward

        # no normalization for now
        if total_visits == 0:
            # edge case: in first step, there is no other state to compare to,
            # so log(1) = 0 and the surprise would be 0, and the max_surprise would be 0 as well
            # making the novelty reward undefined
            # We manually set it to 1.0, meaning that the first state is considered completely novel
            return 1.0
        else:
            # return the novelty as is
            return surprise 

    def calc_latent_intrinsic_reward(self, previous_posterior, current_prior, current_posterior, policy, obs):
        # obs: one step of observation
        # self._current_obs_img will be initialized as zeros
        cur_reward = obs["reward"]
        all_intrinsic_name = self.latents_intrinsic_reward.split("_")

        if self.intr_reward_coeff != 1:
            raise NotImplementedError("Intrinsic reward coefficient must be 1 currently, other values require fixes below.")

        if "sum" in all_intrinsic_name:
            raise NotImplementedError(
                "Sum operation is not supported for latent intrinsic rewards. Use 'mean' instead."
            )
        
        # Only create entries for actual reward components, not operations
        reward_components = [name for name in all_intrinsic_name if name not in ["mean", "mul"]]
        all_intrinsic_rewards = {name: [] for name in reward_components}      

        # Initialize the min-max tracking dictionary if it doesn't exist
        if not hasattr(self, "reward_minmax"):
            self.reward_minmax = {name: {"min": float('inf'), "max": float('-inf')} for name in reward_components}  

        if previous_posterior is None or current_prior is None or current_posterior is None:
            obs["intrinsic_reward"] = np.zeros_like(obs["reward"])
            for name in reward_components:
                obs[f"intrinsic_reward_latent_{name}"] = np.zeros_like(obs["reward"])
                obs[f"intrinsic_reward_latent_{name}_raw"] = np.zeros_like(obs["reward"])
                for key in ["deter", "logit", "stoch"]:
                    # Initialize latents to NaN if they are not available
                    if key in ["stoch", "logit"]:
                        # shape is (num_envs, latent_dim_discrete, latent_dim_discrete) = (8, 32, 32)
                        obs[f"previous_posterior_{key}"] = np.full((self.n_envs, self.z_dim, self.z_dim), np.nan, dtype=np.float32)
                        obs[f"current_posterior_{key}"] = np.full((self.n_envs, self.z_dim, self.z_dim), np.nan, dtype=np.float32)
                        obs[f"current_prior_{key}"] = np.full((self.n_envs, self.z_dim, self.z_dim), np.nan, dtype=np.float32)
                    else:
                        # shape is (num_envs, latent_dim_continuous) = (8, 4096)
                        obs[f"previous_posterior_{key}"] = np.full((self.n_envs, self.h_dim), np.nan, dtype=np.float32)
                        obs[f"current_posterior_{key}"] = np.full((self.n_envs, self.h_dim), np.nan, dtype=np.float32)
                        obs[f"current_prior_{key}"] = np.full((self.n_envs, self.h_dim), np.nan, dtype=np.float32)
            return obs
        
        # save latents for debugging
        # TODO: this is a bit dirty, better to add to logger instead!
        # latents are a dictionary with keys "deter" "logit" and "stoch", want to save them all
        # but we need to save each as a separate entry in the obs dictionary
        for key in current_posterior[0][0].keys():
            obs[f"previous_posterior_{key}"] = previous_posterior[0][0][key].copy()
            obs[f"current_posterior_{key}"] = current_posterior[0][0][key].copy()
            obs[f"current_prior_{key}"] = current_prior[key].copy()

        batch_size = current_posterior[0][0]["stoch"].shape[0]
        post_discrete_states = [
            tuple(
                current_posterior[0][0]["stoch"][i].astype(int).reshape(1, -1).squeeze().tolist()
            )
            for i in range(batch_size)
        ]
        post_continuous_states = current_posterior[0][0]["logit"].astype(np.float32)
        prior_continuous_states = current_prior["logit"].astype(np.float32)

        for name in reward_components:
            if "novelty" in name:
                for i in np.ndindex(cur_reward.shape):
                    # Compute novelty reward based on the state that was just reached
                    cur_state = post_discrete_states[i[0]]
                    if "noveltyperstate" in name:
                        if "factorized" in name:
                            # Compute surprise for each factorized state
                            novelty_reward = 0
                            for j in range(self.z_dim):
                                # select the part of the state that corresponds to the j-th row
                                # of the factorized state by slicing the tuple
                                factorized_state = cur_state[j*self.z_dim:(j+1)*self.z_dim]

                                novelty_reward += self.compute_surprise(
                                    self.factorized_state_counts[j], factorized_state
                                )
                            novelty_reward /= self.z_dim
                        else:
                            novelty_reward = self.compute_surprise(
                                self.state_counts, cur_state
                            )
                            
                    else:
                        tmp_state_counts = self.state_counts.copy()
                        self.update_counts(tmp_state_counts, [cur_state])
                        novelty_reward = self.comp_novelty_entropy(
                            tmp_state_counts
                        )
                        del tmp_state_counts
                        # Update state counts and add novelty reward
                    all_intrinsic_rewards[name].append(novelty_reward)

            if "infogain" in name:
                for i in np.ndindex(cur_reward.shape):
                    post_logit = post_continuous_states[i[0]]
                    post_prob = np.exp(post_logit) / np.sum(
                        np.exp(post_logit), axis=-1, keepdims=True
                    )
                    prior_logit = prior_continuous_states[i[0]]
                    prior_prob = np.exp(prior_logit) / np.sum(
                        np.exp(prior_logit), axis=-1, keepdims=True
                    )
                    kl_divergence = (
                        np.mean(
                            np.sum(post_prob * np.log(post_prob / prior_prob), axis=-1)
                        )
                        * 10
                    )
                    all_intrinsic_rewards[name].append(
                        kl_divergence
                    )

            if "empowerment" in name:
                # Empowerment should be calculated from the current state (z_t)
                empowerment_reward = self.estimate_empowerment_sample(
                    current_posterior, policy, obs
                )
                # TODO: this is not the right place for intr_reward_coeff, need to apply after normalization below
                all_intrinsic_rewards[name] += (
                    empowerment_reward
                ).tolist()

        if "novelty" in self.latents_intrinsic_reward:
            if "factorized" in self.latents_intrinsic_reward:
                # Update counts for each factorized state
                for i in np.ndindex(cur_reward.shape):
                    for j in range(self.z_dim):
                        # select the part of the state that corresponds to the j-th row
                        # of the factorized state by slicing the tuple
                        factorized_state = post_discrete_states[i[0]][
                            j * self.z_dim : (j + 1) * self.z_dim
                        ]
                        self.update_counts(
                            self.factorized_state_counts[j], [factorized_state]
                        )
            else:
                # Update counts for the full state based on the state that was just reached
                self.update_counts(self.state_counts, post_discrete_states)

        for name in reward_components:
            rewards_array = np.array(all_intrinsic_rewards[name])
            
            # Update min and max values for this reward type
            current_min = np.min(rewards_array) if len(rewards_array) > 0 else float('inf')
            current_max = np.max(rewards_array) if len(rewards_array) > 0 else float('-inf')
            
            self.reward_minmax[name]["min"] = min(self.reward_minmax[name]["min"], current_min)
            self.reward_minmax[name]["max"] = max(self.reward_minmax[name]["max"], current_max)
            
            # Store the raw (unnormalized) rewards
            obs[f"intrinsic_reward_latent_{name}_raw"] = rewards_array
            
            # Normalize rewards using min-max scaling if we have valid min and max
            min_val = self.reward_minmax[name]["min"]
            max_val = self.reward_minmax[name]["max"]
            
            if min_val != float('inf') and max_val != float('-inf') and max_val > min_val:
                normalized_rewards = (rewards_array - min_val) / (max_val - min_val)
                obs[f"intrinsic_reward_latent_{name}"] = normalized_rewards
            else:
                # If we don't have valid min-max yet, just use raw values
                obs[f"intrinsic_reward_latent_{name}"] = rewards_array

        if "mul" in self.latents_intrinsic_reward and len(reward_components) >= 2:
            obs["intrinsic_reward"] = np.prod(
                [obs[f"intrinsic_reward_latent_{name}"] for name in reward_components],
                axis=0,
            )
        elif "mean" in self.latents_intrinsic_reward and len(reward_components) >= 2:
            obs["intrinsic_reward"] = np.mean(
                [obs[f"intrinsic_reward_latent_{name}"] for name in reward_components],
                axis=0,
            )
        elif "mul" in self.latents_intrinsic_reward or "mean" in self.latents_intrinsic_reward:
            # Raise an error if only one reward component is specified
            raise ValueError(
                "If using 'mul' or 'mean', at least two reward components must be specified."
            )
        else:
            # If no 'mul' or 'mean' is specified, use the first reward component
            obs["intrinsic_reward"] = obs[f"intrinsic_reward_latent_{reward_components[0]}"]

        obs["reward"] = obs["reward"] + self.intr_reward_coeff * obs["intrinsic_reward"]
        return obs

    def calc_obs_intrinsic_reward(self, obs):
        # obs: one step of observation
        # self._current_obs_img will be initialized as zeros
        cur_reward = obs["reward"]
        all_intrinsic_name = self.obs_intrinsic_reward.split("_")
        all_intrinsic_rewards = {name: [] for name in all_intrinsic_name}

        # Update state counts
        # image_array = obs["image"][0]
        # unique_colors = np.unique(image_array.reshape(-1, image_array.shape[-1]), axis=0)
        # red_tone = [color for color in unique_colors if color[0] > 150 and color[1] < 100 and color[2] < 100]
        if "cbet" in self.obs_intrinsic_reward:
            # Compute difference between next and current state
            img_diff = obs["image"] - self._current_obs_img
            self.update_counts(
                self.change_counts,
                _hash_key(img_diff, self.change_proj, self.change_bias),
            )

        img_states = _hash_key(obs["image"], self.state_proj, self.state_bias)
        # Intialize rewards for intrinsic rewards
        for i in np.ndindex(cur_reward.shape):
            for name in all_intrinsic_name:
                cur_img_state = img_states[i[0]]
                tmp_obs_counts = self.obs_counts.copy()
                self.update_counts(tmp_obs_counts, [cur_img_state])

                if "cbet" in name:
                    img_diff_key_list = _hash_key(
                        [img_diff[i]], self.change_proj, self.change_bias
                    )
                    cbet_reward = self.intr_reward_coeff / (
                        tmp_obs_counts[cur_img_state]
                        + self.change_counts[img_diff_key_list[0]]
                    )
                    all_intrinsic_rewards[name].append(cbet_reward)

                if "novelty" in name:
                    novelty_reward = self.intr_reward_coeff * self.comp_novelty_entropy(
                        tmp_obs_counts
                    )
                    all_intrinsic_rewards[name].append(novelty_reward)

        self.update_counts(self.obs_counts, img_states)
        for name in all_intrinsic_name:
            obs[f"intrinsic_reward_obs_{name}"] = np.array(all_intrinsic_rewards[name])
        obs["intrinsic_reward"] = np.sum(
            [obs[f"intrinsic_reward_obs_{name}"] for name in all_intrinsic_name], axis=0
        )  # TODO: maybe average them?
        obs["reward"] = obs["reward"] + obs["intrinsic_reward"]

        # If random number is less than reset probability, reset the state count
        if np.random.rand() < self.count_reset_prob:
            self.obs_counts.clear()
        if np.random.rand() < self.count_reset_prob:
            self.change_counts.clear()

        return obs

    def update_state_index(self, state_to_index, new_states):
        for new_state in new_states:
            if new_state not in state_to_index.keys():
                state_to_index[new_state] = len(state_to_index)

    def update_transition_counts(self, old_state, new_action, new_next_state):
        old_index = self.state_index[old_state]
        new_index = self.state_index[new_next_state]
        if old_index not in self.transition_counts.keys():
            self.transition_counts[old_state] = defaultdict(lambda: defaultdict(int))
        if new_action not in self.transition_counts[old_state].keys():
            self.transition_counts[old_state][new_action] = defaultdict(int)
        self.transition_counts[old_state][new_action][new_next_state] += 1
        return self.transition_counts
