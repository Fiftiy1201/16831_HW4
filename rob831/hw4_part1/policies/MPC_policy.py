import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 sample_strategy='random',
                 cem_iterations=4,
                 cem_num_elites=5,
                 cem_alpha=1,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

        # Sampling strategy
        allowed_sampling = ('random', 'cem')
        assert sample_strategy in allowed_sampling, f"sample_strategy must be one of the following: {allowed_sampling}"
        self.sample_strategy = sample_strategy
        self.cem_iterations = cem_iterations
        self.cem_num_elites = cem_num_elites
        self.cem_alpha = cem_alpha

        print(f"Using action sampling strategy: {self.sample_strategy}")
        if self.sample_strategy == 'cem':
            print(f"CEM params: alpha={self.cem_alpha}, "
                + f"num_elites={self.cem_num_elites}, iterations={self.cem_iterations}")

    def sample_action_sequences(self, num_sequences, horizon, obs=None):
        if self.sample_strategy == 'random' or (self.sample_strategy == 'cem' and obs is None):
            # Uniform random in [low, high]
            return np.random.uniform(self.low, self.high, size=(num_sequences, horizon, self.ac_dim))

        elif self.sample_strategy == 'cem':
            assert obs is not None, "CEM requires the current observation"

            # ----- shapes & bounds -----
            low  = np.repeat(self.low[None, :],  horizon, axis=0)   # (H, ac_dim)
            high = np.repeat(self.high[None, :], horizon, axis=0)   # (H, ac_dim)

            # Initialize mean/std to cover the whole range
            mean = (low + high) / 2.0                               # (H, ac_dim)
            std  = (high - low) / 2.0                               # (H, ac_dim)
            eps  = 1e-6

            # CEM loop
            for _ in range(self.cem_iterations):
                # Sample K sequences from N(mean, std)
                samples = np.random.randn(num_sequences, horizon, self.ac_dim) * (std[None, :, :] + eps) + mean[None, :, :]
                # Respect action bounds
                samples = np.clip(samples, low[None, :, :], high[None, :, :])   # (K, H, ac_dim)

                # Score sequences with ensemble mean return
                returns = self.evaluate_candidate_sequences(samples, obs)        # (K,)

                # Pick elites
                elite_idx = np.argsort(returns)[-self.cem_num_elites:]           # top-E
                elites    = samples[elite_idx]                                   # (E, H, ac_dim)

                # Update with smoothing
                elite_mean = elites.mean(axis=0)                                  # (H, ac_dim)
                elite_std  = elites.std(axis=0)                                   # (H, ac_dim)
                alpha = getattr(self, "cem_alpha", 0.1)
                mean = alpha * elite_mean + (1 - alpha) * mean
                std  = alpha * elite_std  + (1 - alpha) * std
                # prevent collapse / zero std
                std = np.maximum(std, eps)

            # After refinement, return the mean sequence (clipped)
            best_seq = np.clip(mean, low, high)                                   # (H, ac_dim)
            return best_seq[None, :, :]                                           # (1, H, ac_dim)

        else:
            raise Exception(f"Invalid sample_strategy: {self.sample_strategy}")



    def evaluate_candidate_sequences(self, candidate_action_sequences, obs):
        # TODO(Q2): mean predicted return across ensemble (shape: (N,))
        all_returns = []
        for model in self.dyn_models:
            returns = self.calculate_sum_of_rewards(obs, candidate_action_sequences, model)
            all_returns.append(returns)
        all_returns = np.stack(all_returns, axis=0)   # (E, N)
        mean_returns = np.mean(all_returns, axis=0)   # (N,)
        return mean_returns


    def get_action(self, obs):
        if self.data_statistics is None:
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon, obs=obs)

        if candidate_action_sequences.shape[0] == 1:
            # CEM path (if implemented): single sequence -> take first action
            return candidate_action_sequences[0][0][None]
        else:
            predicted_rewards = self.evaluate_candidate_sequences(candidate_action_sequences, obs)

            # pick best sequence and return its first action
            best_idx = np.argmax(predicted_rewards)
            best_action_sequence = candidate_action_sequences[best_idx]   # (H, ac_dim)
            action_to_take = best_action_sequence[0]                      # (ac_dim,)
            return action_to_take[None]


    def calculate_sum_of_rewards(self, obs, candidate_action_sequences, model):
        """
        Returns per-sequence sum of rewards (shape: [N])
        """
        N = candidate_action_sequences.shape[0]
        sum_of_rewards = np.zeros(N, dtype=np.float32)  # TODO(Q2)

        # batch current obs for all sequences
        current_obs = np.repeat(obs[None, :], repeats=N, axis=0)  # (N, D_obs)

        for t in range(self.horizon):
            actions_t = candidate_action_sequences[:, t, :]  # (N, D_action)

            # env.get_reward may return (rewards, _) or just rewards
            rew_out = self.env.get_reward(current_obs, actions_t)
            rewards = rew_out[0] if isinstance(rew_out, tuple) else rew_out  # (N,)
            sum_of_rewards += rewards.astype(np.float32)

            # rollout one step with the learned dynamics
            next_obs = model.get_prediction(current_obs, actions_t, self.data_statistics)  # (N, D_obs)
            current_obs = next_obs

        return sum_of_rewards



