from torch import nn


class MLP(nn.Module):
    """
    This is a simple MLP model.
    """
    def __init__(self, input_dimension):
        super(MLP, self).__init__()
        self.input_dimension = input_dimension
        self.seq = nn.Sequential(
            nn.Linear(input_dimension, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.seq(x)
