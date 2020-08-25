"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch.nn as nn
import torch.nn.functional as F


class PPO(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(PPO, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.linear = nn.Linear(32 * 4 * 4, 512)
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, num_actions)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, nn.init.calculate_gain('relu'))
                # nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.linear(x.view(x.size(0), -1))
        return self.actor_linear(x), self.critic_linear(x), x

class RandomLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        nn.init.xavier_normal_(self.conv0.weight)

    def forward(self, x):
        # nn.init.xavier_normal_(self.conv0.weight)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        return x

class RandomNetwork(nn.Module):
    """Generic vision network."""

    def __init__(self, in_channels):
        self.in_channels = in_channels

        nn.Module.__init__(self)

        layers = []
        layers.append(RandomLayer(in_channels))
        self._convs = nn.Sequential(*layers)

    def forward(self, obs):
        features = self._hidden_layers(obs)
        return features # NCHW --> NHWC, so the size is the same with the input

    def _hidden_layers(self, obs):
        res = self._convs(obs)  # switch to channel-major
        # res = self._convs(obs)
        # res = res.squeeze(3)
        # res = res.squeeze(2)
        return res

    def re_init(self):
        nn.init.xavier_normal_(self._convs[0].conv0.weight)             