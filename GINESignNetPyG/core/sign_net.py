import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from core.transform import to_dense_list_EVD

import core.model_utils.masked_layers as masked_layers 
from core.model import GNN, RGNN
from core.model_utils.transformer_module import TransformerEncoderLayer, PositionalEncoding 
from core.model_utils.elements import DiscreteEncoder

class GNN3d(nn.Module): 
    """
    Apply GNN on a 3-dimensional data x: n x k x d. 
    Equivalent to apply GNN on k independent nxd 2-d feature.
    * Assume no edge feature for now.
    """
    def __init__(self, n_in, n_out, n_layer, gnn_type='MaskedGINConv'):
        super().__init__()
        self.convs = nn.ModuleList([getattr(masked_layers, gnn_type)(n_in if i==0 else n_out, n_out, bias=False) for i in range(n_layer)])
        self.norms = nn.ModuleList([masked_layers.MaskedBN(n_out) for _ in range(n_layer)])
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(n_in if i==0 else n_out) for i in range(n_layer)]) # only for categorical features

    def reset_parameters(self):
        for conv, norm, edge in zip(self.convs, self.norms, self.edge_encoders): 
            conv.reset_parameters()
            norm.reset_parameters() 
            edge.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr, mask=None):
        # x: n x k x d
        # mask: n x k
        x = x.transpose(0, 1) # k x n x d
        if mask is not None:
            mask = mask.transpose(0, 1) # k x n
        previous_x = 0
        for conv, norm, enc in zip(self.convs, self.norms, self.edge_encoders):
            #TODO: current not work for continuous edge attri
            # x = conv(x, edge_index, enc(edge_attr), mask) # pass mask into
            x = conv(x, edge_index, edge_attr, mask) # pass mask into
            # assert x[~mask].max() == 0 
            if mask is not None:
                x[~mask] = 0
            x = norm(x, mask)
            x = F.relu(x , inplace=False)
            x = x.clone() + previous_x
            previous_x = x.clone()
        return x.transpose(0, 1)

class SetTransformer(nn.Module):
    def __init__(self, nhid, nlayer):
        super().__init__()
        # self.pos_encoder = PositionalEncoding(nhid, freq=100)
        self.pos_encoder = masked_layers.MaskedMLP(1, nhid, nlayer=2)
        
        self.transformer_layers = nn.ModuleList(TransformerEncoderLayer(nhid, n_head=4) for _ in range(nlayer))
        self.out = nn.Sequential(nn.Linear(nhid, nhid, bias=False), nn.BatchNorm1d(nhid))

    def reset_parameters(self):
        # self.transformer_encoder.reset_parameters()
        # self.encoder.reset_parameters()
        self.pos_encoder.reset_parameters()

    def forward(self, x, pos, mask):
        # x: n x k x d
        # pos: n x k 
        # mask: n x k
        # x = self.encoder(x) + self.pos_encoder(pos, mask)
        # x[~mask] = 0
        x = x.clone() + pos
        assert x[~mask].max() == 0
        for layer in self.transformer_layers:
            assert x[~mask].max() == 0
            x,_ = layer(x, mask)
        x = torch.sum(x, dim=1) # n x d #### TODO: change later
        x = self.out(x)
        return x

class SignNet(nn.Module):
    """
        n x k node embeddings => n x n_hid 

        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2):
        super().__init__()
        self.phi = GNN3d(1, n_hid, nl_phi, gnn_type='MaskedGINConv') 
        self.rho = SetTransformer(n_hid, nl_rho)

        self.eigen_encoder1 = masked_layers.MaskedMLP(1, n_hid, nlayer=1)
        self.eigen_encoder2 = masked_layers.MaskedMLP(1, n_hid, nlayer=2)

    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()
        self.eigen_encoder1.reset_parameters()
        self.eigen_encoder2.reset_parameters()

    def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        x = eigV_dense
        # get mask and prepare x
        size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add') # b x 1
        mask = torch.arange(x.size(1), device=x.device)[None, :] < size[:, None]                    # b x N_max
        mask_full = mask[data.batch]

        # transform eigens 
        x = x.unsqueeze(-1)
        # x = self.eigen_encoder1(x, mask_full)
        # assert x[~mask_full].max() == 0
        pos = self.eigen_encoder2(eigS_dense.unsqueeze(-1), mask_full)
        pos = 0 # ignore eigenvalues 

        # phi
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) + self.phi(-x, data.edge_index, data.edge_attr, mask_full)

        # rho = Transformer
        x = self.rho(x, pos=pos, mask=mask_full)

        return x # n x n_hid 

### ORIGINAL SIGNNET #######
class SignNetGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_signnet, nl_gnn):
        # No node/edge features, n_hid=128, n_out=1, nl_signnet=4, nl_gnn=6
        super().__init__()
        self.sign_net = SignNet(n_hid, nl_signnet, nl_rho=1)
        # output of signNet has size Nxn_hid
        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type='SimplifiedPNAConv') #GINEConv

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        pos = self.sign_net(data)
        return self.gnn(data, pos)


class RandomGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_signnet, nl_gnn):
        super().__init__()
        self.gnn = RGNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type='RandPNAConv') #GINEConv

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, data, x_rand):
        return self.gnn(data, x_rand)

