from torch import nn
import torch
from torch import optim
from rob831.hw4_part1.models.base_model import BaseModel
from rob831.hw4_part1.infrastructure.utils import normalize, unnormalize
from rob831.hw4_part1.infrastructure import pytorch_util as ptu


class FFModel(nn.Module, BaseModel):

    def __init__(self, ac_dim, ob_dim, n_layers, size, learning_rate=0.001):
        super(FFModel, self).__init__()

        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.size = size
        self.learning_rate = learning_rate
        self.delta_network = ptu.build_mlp(
            input_size=self.ob_dim + self.ac_dim,
            output_size=self.ob_dim,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.delta_network.to(ptu.device)
        self.optimizer = optim.Adam(
            self.delta_network.parameters(),
            self.learning_rate,
        )
        self.loss = nn.MSELoss()
        self.obs_mean = None
        self.obs_std = None
        self.acs_mean = None
        self.acs_std = None
        self.delta_mean = None
        self.delta_std = None

    def update_statistics(
            self,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        self.obs_mean = ptu.from_numpy(obs_mean)
        self.obs_std = ptu.from_numpy(obs_std)
        self.acs_mean = ptu.from_numpy(acs_mean)
        self.acs_std = ptu.from_numpy(acs_std)
        self.delta_mean = ptu.from_numpy(delta_mean)
        self.delta_std = ptu.from_numpy(delta_std)

    def forward(
            self,
            obs_unnormalized,
            acs_unnormalized,
            obs_mean,
            obs_std,
            acs_mean,
            acs_std,
            delta_mean,
            delta_std,
    ):
        """
        Returns:
            next_obs_pred       = predicted s_{t+1}
            delta_pred_normalized = normalized model output
        """

        # TODO(Q1)
        obs_normalized = normalize(obs_unnormalized, obs_mean, obs_std)
        acs_normalized = normalize(acs_unnormalized, acs_mean, acs_std)

        concatenated_input = torch.cat([obs_normalized, acs_normalized], dim=1)

        # TODO(Q1): model output is normalized delta
        delta_pred_normalized = self.delta_network(concatenated_input)

        # unnormalize delta to get actual delta prediction
        delta_pred = unnormalize(delta_pred_normalized, delta_mean, delta_std)

        # predicted next obs
        next_obs_pred = obs_unnormalized + delta_pred

        return next_obs_pred, delta_pred_normalized

    def get_prediction(self, obs, acs, data_statistics):
        """
        Return predicted next obs as numpy array
        """

        # --- inside get_prediction(...) ---
        obs_t = ptu.from_numpy(obs)
        acs_t = ptu.from_numpy(acs)
        obs_mean = ptu.from_numpy(data_statistics['obs_mean'])
        obs_std  = ptu.from_numpy(data_statistics['obs_std'])
        acs_mean = ptu.from_numpy(data_statistics['acs_mean'])
        acs_std  = ptu.from_numpy(data_statistics['acs_std'])
        delta_mean = ptu.from_numpy(data_statistics['delta_mean'])
        delta_std  = ptu.from_numpy(data_statistics['delta_std'])

        with torch.no_grad():
            next_obs_pred, _ = self(
                obs_t, acs_t,
                obs_mean, obs_std,
                acs_mean, acs_std,
                delta_mean, delta_std
            )

        prediction = ptu.to_numpy(next_obs_pred)  # TODO(Q1)
        return prediction


    def update(self, observations, actions, next_observations, data_statistics):

        # --- inside update(...) ---
        # tensors
        obs_t = ptu.from_numpy(observations)
        acs_t = ptu.from_numpy(actions)
        next_obs_t = ptu.from_numpy(next_observations)

        obs_mean = ptu.from_numpy(data_statistics['obs_mean'])
        obs_std  = ptu.from_numpy(data_statistics['obs_std'])
        acs_mean = ptu.from_numpy(data_statistics['acs_mean'])
        acs_std  = ptu.from_numpy(data_statistics['acs_std'])
        delta_mean = ptu.from_numpy(data_statistics['delta_mean'])
        delta_std  = ptu.from_numpy(data_statistics['delta_std'])

        # normalized target for the model: normalized(s_{t+1} - s_t)
        delta_t = next_obs_t - obs_t
        target = normalize(delta_t, delta_mean, delta_std)                    # TODO(Q1)

        # forward & loss (use only the normalized delta output for loss)
        _, delta_pred_normalized = self(
            obs_t, acs_t,
            obs_mean, obs_std,
            acs_mean, acs_std,
            delta_mean, delta_std
        )
        loss = self.loss(delta_pred_normalized, target)                       # TODO(Q1)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Training Loss': ptu.to_numpy(loss)}


