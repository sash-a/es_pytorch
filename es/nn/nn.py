from torch import nn, Tensor, clamp

from es.utils.ObStat import ObStat


class FullyConnected(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, n_hidden: int, activation: nn.Module,
                 obstat: ObStat, policy_cfg):
        super().__init__()

        layers = [nn.Linear(in_size, hidden_size), activation()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_size, hidden_size), activation()]
        layers += [nn.Linear(hidden_size, out_size)]

        self.model = nn.Sequential(*layers)
        self.action_std = policy_cfg.std
        self.obstat = obstat

    def forward(self, inp: Tensor, **kwargs):
        rs = kwargs['rs']

        inp = clamp((inp - self.obstat.mean) / self.obstat.std, min=-5, max=5)
        a = self.model(inp)
        if self.action_std != 0 and rs is not None:
            a += rs.randn(*a.shape) * self.action_std

        return a.numpy()
