from torch import nn


# TODO gaussian policies
class FullyConnected(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, n_hidden, activation):
        super().__init__()

        layers = [nn.Linear(in_size, hidden_size), activation()]
        for _ in range(n_hidden):
            layers += [nn.Linear(hidden_size, hidden_size), activation()]
        layers += [nn.Linear(hidden_size, out_size), activation()]

        self.model = nn.Sequential(*layers)

    def forward(self, inp):
        return self.model(inp)
