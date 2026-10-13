import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog


class CustomFCNet(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        flat_obs_dim = int(np.prod(obs_space.original_space["obs"].shape))  # (4,4)=16
        style_dim = obs_space.original_space["style"].shape[0]  # 6

        self.layer1 = nn.Linear(flat_obs_dim, 256)
        self.layer2 = nn.Linear(256, 512)
        self.layer3 = nn.Linear(512 + style_dim, 512)
        self.layer4 = nn.Linear(512, 512)



        self.policy_head = nn.Linear(512, num_outputs)
        self.value_head = nn.Linear(512, 1)

        self._last_layer = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]["obs"].float()

        obs = obs.view(obs.size(0), -1)  # (B,16)

        style = input_dict["obs"]["style"].float()      # (B,6)
        # print(style)

        x = torch.tanh(self.layer1(obs))
        x = torch.tanh(self.layer2(x))


        x = torch.cat([x, style], dim=-1)

        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))

        self._last_layer = x
        return self.policy_head(x), state

    def value_function(self):
        return self.value_head(self._last_layer).squeeze(-1)

ModelCatalog.register_custom_model("custom_fcnet", CustomFCNet)
