import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import to_dense_batch
from collections import OrderedDict
from utils.layers import *
 

class GINConvNet(torch.nn.Module):
    def __init__(self, params:dict) -> None:
        """ Constructor.

        Args:
            params (dict): A dictionary containing the parameter to built the graph neural network.

        Required items in params:
            dropout (float): dropout rate.
            batch_size (int): batch size.
            num_features_mol (int): number of molecular features.
            num_features_pro (int): number of features protein.
            mol_output_dim (int): molecule channels during convolution.
            pro_embed_dim (int): protein embedding dimensions.
            pro_len (int): protein sequence len.
            pro_filter (int): number of filters for protein convolution.
            pro_kernel_size (int): protein convolution kernel size. 
            channel_output_dim (int): output dimension of each channel.
            dense_hidden_sizes (list): list of hidden sizes for final dense layers.
        """
        super(GINConvNet, self).__init__()

        self.params = params
        self.pro_len = params.get("pro_len", 85)
        self.num_features_mol = params.get("num_features_mol", 78)
        self.num_features_pro = params.get("num_features_pro", 26)

        # Hyperparameters:
        self.dropout = params.get("dropout", 0.2)
        self.output_dim = params.get("channel_output_dim", 128)
        self.pro_embed_dim = params.get("pro_embed_dim", 128)
        self.mol_dim = params.get("mol_output_dim", 32)
        self.pro_filter = params.get("pro_filter", 32)
        self.pro_kernel = params.get("pro_kernel_size", 8)
        self.output_hidden = params.get("output_hidden_sizes", [1024, 256, 1])

        # Ligand convolution layers
        self.mol_conv1 = molecular_convolutional_layer(self.num_features_mol, self.mol_dim, batch_norm=True)
        self.mol_conv2 = molecular_convolutional_layer(self.mol_dim, self.mol_dim, batch_norm=True)
        self.mol_conv3 = molecular_convolutional_layer(self.mol_dim, self.mol_dim, batch_norm=True)
        self.mol_conv4 = molecular_convolutional_layer(self.mol_dim, self.mol_dim, batch_norm=True)
        self.mol_conv5 = molecular_convolutional_layer(self.mol_dim, self.mol_dim, batch_norm=True)
        #   ligand for output
        self.mol_linear = dense_layer(self.mol_dim, self.output_dim, dropout = self.dropout)

        # Protein layers
        self.pro_embedding = nn.Embedding(self.num_features_pro, self.pro_embed_dim)
        #   protein for output
        self.pro_conv = nn.Conv1d(self.pro_len, self.pro_filter, kernel_size=self.pro_kernel)
        self.pro_linear = dense_layer(self.pro_filter*(self.pro_embed_dim-self.pro_kernel+1), self.output_dim, act_fn= nn.Identity())

        # Output layers
        self.output_linear1 = dense_layer(2*self.output_dim, self.output_hidden[0])
        self.output_linear2 = dense_layer(self.output_hidden[0], self.output_hidden[1], dropout=self.dropout)
        self.output_linear3 = dense_layer(self.output_hidden[1], self.output_hidden[2], act_fn= nn.Identity())  
        


    def forward(self, data_mol, data_seq_1D):
        """Forward pass through the biomodal GINConvNet.

        Args:
            data_mol (torch_geometric.data object)
            data_seq_1D (torch.Tensor): of type int and shape
                `[bs, receptor_sequence_length]`.

        Returns:
            (torch.Tensor): predictions
            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
        """
        # MOLECULE SECTION
        x, edge_index, batch = data_mol.x, data_mol.edge_index, data_mol.batch
        # Molecule convolution
        x = self.mol_conv1(x, edge_index)
        x = self.mol_conv2(x, edge_index)
        x = self.mol_conv3(x, edge_index)
        x = self.mol_conv4(x, edge_index)
        x = self.mol_conv5(x, edge_index)

        x = global_add_pool(x, batch)
        x = self.mol_linear(x)


        # PROTEIN SECTION
        embedded_xt =self.pro_embedding(data_seq_1D)
        xt = self.pro_conv(embedded_xt)
        xt = xt.view(xt.shape[0], -1)
        xt = self.pro_linear(xt)


        x_comb = torch.cat((x, xt), 1)

        # DENSE LAYERS
        x_comb = self.output_linear1(x_comb)
        x_comb = self.output_linear2(x_comb)
        x_out = self.output_linear3(x_comb)


        return x_out

        