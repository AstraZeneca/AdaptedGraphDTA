import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  Sequential, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from collections import OrderedDict


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""
    def forward(self, data):
        return torch.squeeze(data)

def dense_layer(
    input_size, hidden_size, act_fn=nn.ReLU(), batch_norm=False, dropout=0.0
):
    return nn.Sequential(
        OrderedDict(
            [
                ('projection', nn.Linear(input_size, hidden_size)),
                ('batch_norm',
                    nn.BatchNorm1d(hidden_size)
                    if batch_norm
                    else nn.Identity(),
                ),
                ('act_fn', act_fn),
                ('dropout', nn.Dropout(p=dropout)),
            ]
        )
    )


def molecular_convolutional_layer(
    input_channels,
    output_channels,
    act_fn=nn.ReLU(),
    batch_norm=False,
):
    """Molecular Convolutional layer.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
        act_fn (callable): Functional of the nonlinear activation.
        batch_norm (bool): whether batch normalization is applied.
        dropout (float): Probability for each input value to be 0.

    Returns:
        callable: a function that can be called with inputs.
    """
    linear_layers = nn.Sequential(
        nn.Linear(input_channels,output_channels ), 
        act_fn, 
        nn.Linear(output_channels,output_channels)
        )

    return Sequential('x, edge_index',
            [
                ( GINConv(linear_layers), 'x, edge_index -> x2'),
                (
                    nn.BatchNorm1d(output_channels) if batch_norm else nn.Identity(),
                    'x2 -> x3'
                ),
            ]
    )

