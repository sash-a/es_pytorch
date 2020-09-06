from torch import nn


class FullyConnected(nn.Module):
    def __init__(self, in_size: int, out_size: int, hidden_size: int, n_hidden: int, activation: nn.Module, policy_cfg):
        super().__init__()

        layers = [nn.Linear(in_size, hidden_size), activation()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_size, hidden_size), activation()]
        layers += [nn.Linear(hidden_size, out_size)]

        self.model = nn.Sequential(*layers)
        self.std = policy_cfg.std

    def forward(self, inp, **kwargs):
        rs = kwargs['rs']

        a = self.model(inp)
        if self.std != 0 and rs is not None:
            a += rs.randn(*a.shape) * self.std

        return a.numpy()
