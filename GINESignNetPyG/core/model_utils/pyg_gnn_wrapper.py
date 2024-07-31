import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from core.model_utils.elements import MLP
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops

class GINConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias, with_norm=False) ##### Do not use BN!!!!
        self.layer = gnn.GINConv(self.nn, train_eps=True)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout//nhead, nhead, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)

class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        # self.nn = MLP(nin, nout, 2, False, bias=bias)
        # self.layer = gnn.GCNConv(nin, nin, bias=True)
        self.layer = gnn.GCNConv(nin, nout, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)
        # return self.nn(F.relu(self.layer(x, edge_index)))

from torch_scatter import scatter
from torch_geometric.utils import degree
class SimplifiedPNAConv(gnn.MessagePassing):
    def __init__(self, nin, nout, bias=True, aggregators=['mean', 'min', 'max', 'std'], data_max_degree=200, **kwargs): # ['mean', 'min', 'max', 'std'],
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.aggregators = aggregators
        self.pre_nn = MLP(3*nin, nin, 2, False)
        self.post_nn = MLP((len(aggregators) + 1 +1) * nin, nout, 2, False, bias=bias)
        # self.post_nn = MLP((len(aggregators) + 1 ) * nin, nout, 2, False)
        self.deg_embedder = nn.Embedding(200, nin) 

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    # x is Nx128(i.e. num_node_features), edge_index is 2xnum_edges, edge_attr is num_edgesx128(i.e. num_edge_features)
    def forward(self, x, edge_index, edge_attr):
        import pdb; pdb.set_trace()
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, out], dim=-1)
        out = self.post_nn(out)
        # return x + out
        return out

    # x_i and x_j are num_edges x 128
    def message(self, x_i, x_j, edge_attr):
        import pdb; pdb.set_trace()
        if edge_attr is not None:
            #h is num_edgesx(2*num_node_features + num_edge_features)
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return self.pre_nn(h)

    # inputs is num_edgesx nhid (num_node_features==edge_features)
    def aggregate(self, inputs, index, dim_size=None):
        import pdb; pdb.set_trace()
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu_(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')  
            outs.append(out)

        outs.append(self.deg_embedder(degree(index, dim_size, dtype=index.dtype)))
        out = torch.cat(outs, dim=-1)

        return out

def reconstruct_laplacian_from_flat(laplacian, num_nodes_per_graph):
    """
    Reconstruct the Laplacian matrices from the flattened form and
    create a block-diagonal matrix for a batched operation.
    """
    batch_size = len(num_nodes_per_graph)
    max_num_nodes = max(num_nodes_per_graph)
    laplacian_matrices = []
    offset = 0
    for nodes in num_nodes_per_graph:
        L_flat = laplacian[offset:offset + nodes * nodes]
        L = L_flat.view(nodes, nodes)
        # Create a padded Laplacian matrix with the shape (max_num_nodes, max_num_nodes)
        L_padded = torch.zeros((max_num_nodes, max_num_nodes), device=laplacian.device)
        L_padded[:nodes, :nodes] = L
        laplacian_matrices.append(L_padded)
        offset += nodes * nodes

    # Stack the padded Laplacian matrices to create a batch
    laplacian_batch = torch.stack(laplacian_matrices)
    return laplacian_batch

'''
    PNA Conv layer for random input.
    NOTE: If you use a different dataset than ZINC, have to change data_max_deg param here.
'''
class RandPNAConv(gnn.MessagePassing):
    def __init__(self, nin, nout, bias=True, aggregators=['mean', 'min', 'max', 'std'], bn=False, pna_layers=2,
                 data_max_deg=200, **kwargs): # used to be mean only? # ['mean', 'min', 'max', 'std'],
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.aggregators = aggregators
        self.pre_nn = MLP(2*nin, nin, pna_layers, False, with_norm=False) # with_norm=bn
        self.post_nn = MLP((len(aggregators) + 2) * nin, nout, pna_layers, False, with_norm=bn, bias=bias)
        self.deg_embedder = nn.Embedding(data_max_deg, nin) 

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    # Completely Random samples only - edge_attr and laplacian are None
    # x has shape NxMxD, edge_index is 2xnum_edges. Thus num_node_dim=1 here
    def forward(self, x, edge_index, edge_attr, laplacian=None):
        # Pass in a dummy variable for edge_attr if its None#
        num_edges = edge_index.size(1)
        edge_feature_dim = 1
        if edge_attr is None:
            edge_attr = torch.zeros((num_edges, edge_feature_dim), dtype=torch.float32, device=x.device)

        if laplacian is not None:
            edge_attr = torch.cat((edge_attr, laplacian.unsqueeze(-1)), dim=-1)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
            laplacian = edge_attr[:,-1]
            edge_attr = edge_attr[:,:-1]
        out = self.propagate(edge_index, x=x.float(), edge_attr=edge_attr, laplacian=laplacian) # out has shape ExMx5*nhid
        out = torch.cat([x, out], dim=-1)
        out = self.post_nn(out)
        # return x + out
        return out

    # x_i and x_j have shape num_edgesxMxD
    def message(self, x_i, x_j, edge_attr, laplacian=None):
        if laplacian is not None:
            L = laplacian.shape[0]
            if edge_attr is not None:
                h = torch.cat([x_i, x_j * laplacian.view(L, 1, 1), edge_attr.unsqueeze(-1).expand(-1, -1, x_i.shape[-1])], dim=1)
                A, H, B = h.shape
            else:
                h = torch.cat([x_i, x_j], dim=-1)
        output = torch.cat((x_i, x_j), dim=-1)
        output = self.pre_nn(output) # shape num_edges*Mxnhid
        return output #.view(E, M, -1)

    # inputs has shape ExMxnhid, and index has shape E,
    def aggregate(self, inputs, index, dim_size=None):
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu_(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')  
            outs.append(out)
        outs.append(self.deg_embedder(degree(index, dim_size, dtype=index.dtype)).unsqueeze(1).expand(-1, inputs.shape[-2], -1)) # degree has shape N,
        # deg embedder --> Nx128 (NxD)
        # expanded to have Nx128xM
        out = torch.cat(outs, dim=-1)
        return out
    
'''
    This is what RGNN_GRAPH uses. Since the model doesn't use random input, this model is pretty much the same as
    SimplifiedPNAConv, except it can use the laplacian w/ added self loops.
'''
class RandPNAConv_GRAPH(gnn.MessagePassing):
    def __init__(self, nin, nout, bias=True, aggregators=['mean', 'min', 'max', 'std'], bn='layernorm', pna_layers=2, laplacian=False, **kwargs): # used to be mean only? # ['mean', 'min', 'max', 'std'],
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.aggregators = aggregators
        if laplacian:
            self.pre_nn = MLP(3*nin+1, nin, 2, False)#MLP(2*nin+128+1, nin, 2, False)
        else:
            self.pre_nn = MLP(3*nin, nin, 2, False)
        self.post_nn = MLP((len(aggregators) + 1 +1) * nin, nout, 2, False, bias=bias)
        # self.post_nn = MLP((len(aggregators) + 1 ) * nin, nout, 2, False)
        self.deg_embedder = nn.Embedding(200, nin)
        self.nin=nin 

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    def forward(self, x, edge_index, edge_attr, laplacian=None):
        if laplacian is not None:
            edge_attr = torch.cat((edge_attr, laplacian.unsqueeze(-1)), dim=-1)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr=edge_attr, num_nodes=x.size(0))
            laplacian = edge_attr[:,-1]
            edge_attr = edge_attr[:,:-1]
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, laplacian=laplacian)
        out = torch.cat([x, out], dim=-1)
        out = self.post_nn(out)
        # return x + out
        return out

    def message(self, x_i, x_j, edge_attr, laplacian):
        if edge_attr is not None:
            if laplacian is not None:
                L = laplacian.shape[0]
                h = torch.cat([x_i, x_j, edge_attr, laplacian.view(L, 1)], dim=-1)
            else:
                h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return self.pre_nn(h)

    def aggregate(self, inputs, index, dim_size=None):
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu_(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')  
            outs.append(out)

        outs.append(self.deg_embedder(degree(index, dim_size, dtype=index.dtype)))
        out = torch.cat(outs, dim=-1)

        return out
