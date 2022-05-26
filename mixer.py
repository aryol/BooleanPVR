# adapted from lucidrains/mlp-mixer-pytorch (https://github.com/lucidrains/mlp-mixer-pytorch)
# See https://arxiv.org/abs/2105.01601 for information about MLP-Mixer
from torch import nn
from functools import partial
from einops.layers.torch import Reduce


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


class TokenMixer(nn.Module):
    def __init__(self, *, seq_len, output_dim, dim, depth, expansion_factor=4, 
            expansion_factor_token=0.5, dropout=0., inputs_are_pos_neg_ones=True):
        super().__init__()
        self._inputs_are_pos_neg_ones = inputs_are_pos_neg_ones
        chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
        self.body = nn.Sequential(
                nn.Embedding(num_embeddings=2, embedding_dim=dim),
                *[nn.Sequential(
                    PreNormResidual(dim, FeedForward(seq_len, expansion_factor, dropout, chan_first)),
                    PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
                    ) for _ in range(depth)],
                nn.LayerNorm(dim),
                Reduce('b n c -> b c', 'mean'),
                nn.Linear(dim, output_dim))

    def forward(self, inputs):
        if self._inputs_are_pos_neg_ones:
            # convert from +1/-1 to 0/1
            inputs = (inputs + 1) / 2
        inputs = inputs.int()
        return self.body(inputs)
