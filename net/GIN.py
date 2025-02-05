import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINConv as GIN_no_edge_attr
from torch_geometric.nn import GCNConv as GCN_no_edge_attr
#from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.utils import degree

from torch_geometric.nn.models import GAT,MLP,GCN,GraphSAGE,GAE,GIN

### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")
        

        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        #self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.bond_encoder = nn.Linear(13,emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim):
        super(GCNConv, self).__init__(aggr='add')
        
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        #self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        self.bond_encoder = nn.Linear(13,emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index

        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

### GNN to generate node embedding
class GNN_node(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, input_dim, num_layer, emb_dim, edge_attr_option = True, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        
        super(GNN_node, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio

        self.residual = residual
        self.edge_attr_option = edge_attr_option
        self.JK=JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        #self.atom_encoder = AtomEncoder(emb_dim)
        self.atom_encoder = nn.Linear(input_dim, emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            if gnn_type == 'gin':
                if self.edge_attr_option:
                    self.convs.append(GINConv(emb_dim))
                else:
                    self.convs.append(GIN_no_edge_attr(emb_dim))
            elif gnn_type == 'gcn':
                if self.edge_attr_option:
                    self.convs.append(GCNConv(emb_dim))
                else:
                    self.convs.append(GCN_no_edge_attr(emb_dim, emb_dim))
            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward( self, batched_data ):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        #print(x.shape)
        #x = x.clone().float()
        x = x.float()
        if self.edge_attr_option:
            edge_attr = edge_attr.clone().float()
        
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        
        for layer in range(self.num_layer):
            if self.edge_attr_option:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation
    
#revised by yunpeng xia
class GNN_node_test(torch.nn.Module):
    """
    Output:
        node representations
    
    Revised by Yunpeng Xia.

    """
    def __init__(self, input_dim, num_layer, emb_dim, edge_attr_option = True, drop_ratio = 0.5, JK = "last", residual = False, gnn_type = 'gin'):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
            drop_ratio (float): dropout rate`   
            JK (str): one of 'last', 'sum', 'max', 'attention'  
            residual (bool): whether to add residual connection 
            edge_attr_option (bool): whether to use edge attributes
            gnn_type (str): select from 'gin', 'gcn', 'gat', 'graphsage' and 'mlp'
        '''
        super(GNN_node_test, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.residual = residual
        self.gnn_type = gnn_type

        ### add residual connection or not
        self.edge_attr_option = edge_attr_option

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        
        self.atom_encoder = nn.Linear(input_dim, emb_dim)

        for layer in range(num_layer):
            if gnn_type == 'gin':
                self.convs.append(GIN(in_channels=emb_dim,hidden_channels=emb_dim,num_layers=num_layer))
            elif gnn_type == 'gcn':
                self.convs.append(GCN(in_channels=emb_dim,hidden_channels=emb_dim,num_layers=num_layer))
            elif gnn_type == 'gat':
                self.convs.append(GAT(in_channels=emb_dim,hidden_channels=emb_dim,num_layers=num_layer))
            elif gnn_type == 'mlp':
                self.convs.append(MLP([emb_dim,emb_dim]))
            elif gnn_type == 'graphsage':
                self.convs.append(GraphSAGE(in_channels=emb_dim,hidden_channels=emb_dim,num_layers=num_layer))

            else:
                raise ValueError('Undefined GNN type called {}'.format(gnn_type))

            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward( self, batched_data ):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        #print(x.shape)
        #x = x.clone().float()
        x = x.float()
        if self.edge_attr_option:
            edge_attr = edge_attr.clone().float()
        
        ### computing input node embedding
        h_list = [self.atom_encoder(x)]
        
        for layer in range(self.num_layer):
            if self.gnn_type=='mlp':
                h = self.convs[layer](h_list[layer])
            elif self.edge_attr_option:
                h = self.convs[layer](h_list[layer], edge_index, edge_attr = edge_attr)
            else:
                h = self.convs[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = 0
            for layer in range(self.num_layer + 1):
                node_representation += h_list[layer]
        return node_representation