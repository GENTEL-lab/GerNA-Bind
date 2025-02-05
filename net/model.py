import torch
import pickle
import sys
#sys.path.append("net/")
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from equiformer_pytorch import Equiformer
from torch_geometric.data import Batch
from net.GIN import GNN_node,GNN_node_test
from net.MLP import MLP
from net.Trigonometry import Transition,TriangleProteinToCompound_v2,TriangleSelfAttentionRowWise,TriangleProteinToCompound_v3
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
import sys
#sys.path.append("../")
from utils.net_utils import *
import time
from edl_pytorch import Dirichlet, NormalInvGamma, evidential_classification

class GerNA(nn.Module):
    def __init__(self, params, input_dim_rna = 640, input_dim_mol = 55, trigonometry = True, mol_graph = True, coors = True, rna_repre = True, rna_graph = True, coors_3_bead = True, uncertainty = False, num_classes = 2, use_other_GNN = False, GNN_type = "gcn"):
        super(GerNA, self).__init__()
        
        self.dropout_layer = nn.Dropout(0.4)
        """hyper part"""
        # GNN_depth, inner_CNN_depth, DMA_depth, k_head, kernel_size, hidden_size1, hidden_size2 = params
        GNN_depth, DMA_depth, hidden_size1, hidden_size2 = params
        self.GNN_depth = GNN_depth
        #self.inner_CNN_depth = inner_CNN_depth
        self.DMA_depth = DMA_depth
        #self.k_head = k_head
        #self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.input_dim_rna = input_dim_rna
        #if not embedding
        self.input_dim_mol = input_dim_mol
        self.trigonometry = trigonometry
        self.mol_graph = mol_graph
        self.coors = coors
        self.rna_graph = rna_graph
        self.coors_3_bead = coors_3_bead
        self.rna_repre = rna_repre
        self.uncertainty = uncertainty
        self.num_classes = num_classes
        self.use_other_GNN = use_other_GNN
        self.GNN_type = GNN_type
        
        
        """GCN module for molecule and RNA"""
        if self.rna_graph:
            self.GCN_rna = GNN_node(self.input_dim_rna, self.GNN_depth, self.hidden_size1, edge_attr_option = False, gnn_type = 'gcn')  #RNA edge has no attribute features
        
        if self.mol_graph:
            if use_other_GNN:
                self.GCN_mol = GNN_node_test(self.input_dim_mol, self.GNN_depth, self.hidden_size1, gnn_type = GNN_type)
            else:
                self.GCN_mol = GNN_node(self.input_dim_mol, self.GNN_depth, self.hidden_size1, gnn_type = 'gcn')

        """MLP Module"""
        if self.rna_repre:
            self.mlp_rna = MLP(self.input_dim_rna, self.hidden_size1)
        
        """EquiFormer Block"""
        if self.coors:
            self.equi_mol = Equiformer( num_tokens = 10, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0, 
                                        num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)
            if self.coors_3_bead:
                self.equi_rna = Equiformer( num_tokens = 5, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0, 
                                           num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)
            else:
                self.equi_rna = Equiformer( num_tokens = 9, dim = (16, 4, 2), dim_head = (10, 10, 10), heads = (2, 2, 2), num_linear_attn_heads = 0, 
                                           num_degrees = 3, depth = 2, attend_self = True, l2_dist_attention = False, reversible = True)

        """pairwise_prediction_module"""
        if self.mol_graph:
            self.pairwise_mol_2d = nn.Linear(self.hidden_size1, self.hidden_size1)
        if self.rna_graph:
            self.pairwise_rna_2d = nn.Linear(self.hidden_size1, self.hidden_size1)
        if self.rna_repre:
            self.pairwise_rna_1d = nn.Linear(self.hidden_size1, self.hidden_size1)
        if self.coors:
            self.pairwise_rna_3d = nn.Linear(16, 16)
            self.pairwise_mol_3d = nn.Linear(16, 16)
            self.layernorm_3d = torch.nn.LayerNorm(16)
            
        if trigonometry:
            """Tankbind gengxin pair"""
            self.n_trigonometry_module_stack = 2 # n_trigonometry_module_stack
            if self.mol_graph:

                self.rna_to_mol_list_2d = []
                self.rna_to_mol_list_2d = nn.ModuleList([TriangleProteinToCompound_v3(embedding_channels=self.hidden_size1, c=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                self.triangle_self_attention_list_2d = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                self.tranistion_2d = Transition(embedding_channels=self.hidden_size1, n=4)
                self.layernorm_2d = torch.nn.LayerNorm(self.hidden_size1)
                self.dropout_2d = nn.Dropout2d(p=0.25)
                self.linear_2d = nn.Linear(self.hidden_size1, 1)
                self.rna_pair_embedding_2d = nn.Linear(16,self.hidden_size1)
                self.mol_pair_embedding_2d = nn.Linear(16,self.hidden_size1)
            #Origin, which use feat to represent the coords.
                # self.rna_to_mol_list = []
                # self.rna_to_mol_list = nn.ModuleList([TriangleProteinToCompound_v2(embedding_channels=self.hidden_size1, c=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                # self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=self.hidden_size1) for _ in range(self.n_trigonometry_module_stack)])
                # self.tranistion = Transition(embedding_channels=self.hidden_size1, n=4)
                # self.layernorm = torch.nn.LayerNorm(self.hidden_size1)
                # self.dropout = nn.Dropout2d(p=0.25)
                # self.linear = nn.Linear(self.hidden_size1, 1)

            if self.coors:
                self.rna_to_mol_list_3d = []
                self.rna_to_mol_list_3d = nn.ModuleList([TriangleProteinToCompound_v3(embedding_channels=16, c=16) for _ in range(self.n_trigonometry_module_stack)])
                self.triangle_self_attention_list_3d = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=16) for _ in range(self.n_trigonometry_module_stack)])
                self.tranistion_3d = Transition(embedding_channels=16, n=4)
                self.layernorm_3d = torch.nn.LayerNorm(16)
                self.dropout_3d = nn.Dropout2d(p=0.25)
                self.linear_3d = nn.Linear(16, 1)
                self.rna_pair_embedding = nn.Linear(16,16)
                self.mol_pair_embedding = nn.Linear(16,16)
            
        """Affinity Prediction Module"""
        if self.mol_graph:
            self.mol_2d_final = nn.Linear( self.hidden_size1, self.hidden_size2 )
            if self.rna_repre:
                self.rna_1d_final = nn.Linear( self.hidden_size1, self.hidden_size2 )
            if self.rna_graph:
                self.rna_2d_final = nn.Linear( self.hidden_size1, self.hidden_size2 )
        if self.coors:
            self.mol_3d_final = nn.Linear( 16, self.hidden_size2 )
            self.rna_3d_final = nn.Linear( 16, self.hidden_size2 )
        #Output layer
        if self.mol_graph and self.coors:
            if uncertainty:
                self.W_out = nn.Sequential(
                    nn.Linear(2*self.hidden_size2*self.hidden_size2, 1024),
                    nn.ReLU(),nn.Dropout(0.4),
                    nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), Dirichlet(128, self.num_classes))
            else:
                self.W_out = nn.Sequential(
                    nn.Linear(2*self.hidden_size2*self.hidden_size2, 1024),
                    nn.ReLU(),nn.Dropout(0.4),
                    nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), nn.Linear(128, 1))
        else:
            if uncertainty:
                self.W_out = nn.Sequential(
                    nn.Linear(self.hidden_size2*self.hidden_size2, 1024),
                    nn.ReLU(),nn.Dropout(0.4),
                    nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), Dirichlet(128, self.num_classes))
                # self.W_out = nn.Sequential(  #change to regression
                #     nn.Linear(self.hidden_size2*self.hidden_size2, 1024),
                #     nn.ReLU(),nn.Dropout(0.4),
                #     nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), NormalInvGamma(128, 1))
            else:
                self.W_out = nn.Sequential(
                    nn.Linear(self.hidden_size2*self.hidden_size2, 1024),
                    nn.ReLU(),nn.Dropout(0.4),
                    nn.Linear(1024, 128),nn.ReLU(),nn.Dropout(0.4), nn.Linear(128, 1))

        #DMA parameters
        if self.mol_graph:
#             self.mc0 = nn.Linear(hidden_size2, hidden_size2)
#             self.mp0 = nn.Linear(hidden_size2, hidden_size2)
            self.mc1 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.mp1 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc0 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hp0 = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.hp1 = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.m_to_r_transform = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.r_to_m_transform = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.GRU_dma = nn.GRUCell(self.hidden_size2, self.hidden_size2)
            
        if self.coors:
#             self.mc0_3d = nn.Linear(hidden_size2, hidden_size2)
#             self.mp0_3d = nn.Linear(hidden_size2, hidden_size2)
            self.mc1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.mp1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc0_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hp0_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.hc1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.hp1_3d = nn.ModuleList([nn.Linear(self.hidden_size2, 1) for i in range(self.DMA_depth)])
            self.m_to_r_transform_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.r_to_m_transform_3d = nn.ModuleList([nn.Linear(self.hidden_size2, self.hidden_size2) for i in range(self.DMA_depth)])
            self.GRU_dma_3d = nn.GRUCell(self.hidden_size2, self.hidden_size2)

#         """Pairwise Interaction Prediction Module"""
#         self.pairwise_molecule = nn.Linear(self.hidden_size1, self.hidden_size1)
#         self.pairwise_rna = nn.Linear(self.hidden_size1, self.hidden_size1)
    
    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax
    
    def Pairwise_pred_module(self, batch_size, RNA_1d_fea=None, RNA_2d_fea=None, Mol_2d_fea=None, seq_mask=None, vertex_mask=None, RNA_feats=None, 
                             Mol_feats=None, RNA_C4_coors= None, RNA_coors=None, Mol_coors=None, RNA_mask=None, Mol_mask=None, Mol_LAS_dis=None):

        if self.mol_graph:
            pairwise_mol_2d_fea = F.leaky_relu(self.pairwise_mol_2d(Mol_2d_fea), 0.1)
            if self.rna_repre:
                pairwise_rna_2d_fea = F.leaky_relu(self.pairwise_rna_1d(RNA_1d_fea), 0.1)
                if self.rna_graph:
                    self.fusion_grad_frac = 0.5
                    pairwise_rna_2d_fea = self.fusion_grad_frac * F.leaky_relu(self.pairwise_rna_2d(RNA_2d_fea), 0.1) + (1-self.fusion_grad_frac) * pairwise_rna_2d_fea 
            else:
                pairwise_rna_2d_fea = F.leaky_relu(self.pairwise_rna_2d(RNA_2d_fea), 0.1)
        if self.coors:
            pairwise_rna_3d_fea = F.leaky_relu(self.pairwise_rna_3d( RNA_feats ), 0.1)
            pairwise_mol_3d_fea = F.leaky_relu(self.pairwise_mol_3d( Mol_feats ), 0.1)
        
        #revised in 20240902
        #special_pair_2d = pairwise_rna_2d_fea

        if not self.trigonometry:
            if self.coors: #计算3d_interaction matrix
                pairwise_pred_3d = torch.sigmoid(torch.matmul(pairwise_rna_3d_fea, pairwise_mol_3d_fea.transpose(1,2)))
                pairwise_mask_3d = torch.matmul( RNA_mask.view(batch_size,-1,1), Mol_mask.view(batch_size,1,-1) )
                pairwise_pred_3d = pairwise_pred_3d * pairwise_mask_3d
            if self.mol_graph:
                #2d_interaction_matrix  pairwise_pred_2d
                pairwise_pred_2d = torch.sigmoid(torch.matmul(pairwise_rna_2d_fea, pairwise_mol_2d_fea.transpose(1,2)))
                pairwise_mask_2d = torch.matmul( seq_mask.view(batch_size,-1,1), vertex_mask.view(batch_size,1,-1) )
                pairwise_pred_2d = pairwise_pred_2d * pairwise_mask_2d
                if self.coors:
                    return torch.sigmoid(pairwise_pred_2d), torch.sigmoid(pairwise_pred_3d)
                else:
                    return torch.sigmoid(pairwise_pred_2d)
            else:
                return torch.sigmoid(pairwise_pred_3d)
        else: #The pairwise interaction prediction which have trigonometry-aware constraint.
            if self.mol_graph:
                rna_pair_2d = get_pair_dis_one_hot(RNA_C4_coors, bin_size=2, bin_min=-1, bin_max=30)
                mol_pair_2d = get_mol_pair_dis_distribution(Mol_coors, Mol_LAS_dis)
                
                rna_pair_2d = self.rna_pair_embedding_2d(rna_pair_2d.float())
                mol_pair_2d = self.mol_pair_embedding_2d(mol_pair_2d.float())  #rna_pair_embedding_2d and mol_pair_embedding_2d need re-define.
                
                mol_out_batched_2d = self.layernorm_2d(Mol_2d_fea)
                
                if self.rna_repre:
                    rna_out_batched_1d = self.layernorm_2d(RNA_1d_fea)
                    if self.rna_graph:
                        rna_out_batched_2d = self.layernorm_2d(RNA_2d_fea)
                        self.fusion_grad_frac = 0.5
                        rna_out_batched_2d = self.fusion_grad_frac * rna_out_batched_2d + (1-self.fusion_grad_frac) * rna_out_batched_1d
                    else:
                        rna_out_batched_2d = rna_out_batched_1d
                else:
                    rna_out_batched_2d = self.layernorm_2d(RNA_2d_fea)

                z_2d = rna_out_batched_2d.unsqueeze(2) * mol_out_batched_2d.unsqueeze(1)
                z_mask_2d = seq_mask.unsqueeze(2) * vertex_mask.unsqueeze(1)

                for _ in range(1):
                    for i_module in range(self.n_trigonometry_module_stack):
                        z_2d = z_2d + self.dropout_2d(self.rna_to_mol_list_2d[i_module](z_2d, rna_pair_2d, mol_pair_2d, z_mask_2d.unsqueeze(-1)))
                        z_2d = z_2d + self.dropout_2d(self.triangle_self_attention_list_2d[i_module](z_2d, z_mask_2d))
                        z_2d = self.tranistion_2d(z_2d)
                pairwise_pred_2d = self.linear_2d(z_2d).squeeze(-1)  #linear 2d also need to re-define
                pairwise_pred_2d = pairwise_pred_2d * z_mask_2d #[z_mask.long()]

            if self.coors:
                rna_pair = get_pair_dis_one_hot(RNA_coors, bin_size=2, bin_min=-1, bin_max=30)
                mol_pair = get_mol_pair_dis_distribution(Mol_coors, Mol_LAS_dis)
                                                   
                rna_pair = self.rna_pair_embedding(rna_pair.float())
                mol_pair = self.mol_pair_embedding(mol_pair.float())
                
                mol_out_batched_3d = self.layernorm_3d(Mol_feats)
                rna_out_batched_3d = self.layernorm_3d(RNA_feats)
                # z of shape, b, protein_length, compound_length, channels.
                # z = torch.einsum("bik,bjk->bijk", rna_out_batched_3d, mol_out_batched_3d)
                # z_mask = torch.einsum("bi,bj->bij", RNA_mask, Mol_mask)

                z = rna_out_batched_3d.unsqueeze(2) * mol_out_batched_3d.unsqueeze(1)
                z_mask = RNA_mask.unsqueeze(2) * Mol_mask.unsqueeze(1)

                for _ in range(1):
                    for i_module in range(self.n_trigonometry_module_stack):
                        z = z + self.dropout_3d(self.rna_to_mol_list_3d[i_module](z, rna_pair, mol_pair, z_mask.unsqueeze(-1)))  #这边用到坐标上可能会报错
                        z = z + self.dropout_3d(self.triangle_self_attention_list_3d[i_module](z, z_mask))
                        z = self.tranistion_3d(z)
                pairwise_pred_3d = self.linear_3d(z).squeeze(-1)
                pairwise_pred_3d = pairwise_pred_3d * z_mask #[z_mask.long()]
                if self.mol_graph:
                    return torch.sigmoid(pairwise_pred_2d), torch.sigmoid(pairwise_pred_3d)
                else:
                    return torch.sigmoid(pairwise_pred_3d)
            else:
                return torch.sigmoid(pairwise_pred_2d)
                                                   

    def Affinity_pred_module(self, batch_size, RNA_1d_fea=None, RNA_2d_fea=None, Mol_2d_fea=None, seq_mask=None, vertex_mask=None, RNA_feats=None, Mol_feats=None, RNA_mask=None, Mol_mask=None, pairwise_pred_2d=None, pairwise_pred_3d=None ): 
        if self.mol_graph:
            Mol_2d_fea = F.leaky_relu(self.mol_2d_final(Mol_2d_fea), 0.1)
            if self.rna_repre:
                RNA_1d_fea = F.leaky_relu(self.rna_1d_final(RNA_1d_fea), 0.1)
                if self.rna_graph:
                    RNA_2d_fea = F.leaky_relu(self.rna_2d_final(RNA_2d_fea), 0.1)
                    self.fusion_grad_frac = 0.5
                    RNA_2d_fea = self.fusion_grad_frac * RNA_2d_fea + ( 1 - self.fusion_grad_frac ) * RNA_1d_fea
                else:
                    RNA_2d_fea = RNA_1d_fea
            else:
                RNA_2d_fea = F.leaky_relu(self.rna_2d_final(RNA_2d_fea), 0.1)
        if self.coors:
            RNA_feats = F.leaky_relu(self.rna_3d_final(RNA_feats), 0.1)
            Mol_feats = F.leaky_relu(self.mol_3d_final(Mol_feats), 0.1)
        
        if self.coors:
            mf_3d, rf_3d = self.dma_gru_3d( batch_size, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_3d )  # RNA_length * Mol_length
            kroneck_3d = F.leaky_relu(torch.matmul(mf_3d.unsqueeze(-1), rf_3d.unsqueeze(-2)).view(-1,self.hidden_size2*self.hidden_size2), 0.1)
        if self.mol_graph:
            mf_2d, rf_2d = self.dma_gru_2d( batch_size, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, pairwise_pred_2d )   # RNA_length * Mol_length
            kroneck_2d = F.leaky_relu(torch.matmul(mf_2d.view(batch_size,-1,1), rf_2d.view(batch_size,1,-1)).view(batch_size,-1), 0.1)
        #print(kroneck_2d.shape, kroneck_3d.shape)
        if self.mol_graph and self.coors:
            kroneck = torch.cat((kroneck_2d,kroneck_3d),dim=1)
        elif self.mol_graph:
            kroneck = kroneck_2d
        elif self.coors:
            kroneck = kroneck_3d
        if self.uncertainty:
            affinity_pred = self.W_out(kroneck)
            #affinity_pred = torch.sigmoid( self.W_out(kroneck) )
        else:
            affinity_pred = torch.sigmoid( self.W_out(kroneck) )
        return affinity_pred

    def dma_gru_2d(self, batch_size, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, pairwise_pred_2d ):
        
        vertex_mask = vertex_mask.view(batch_size,-1,1)
        seq_mask = seq_mask.view(batch_size,-1,1)
        m0 = torch.sum(Mol_2d_fea * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        r0 = torch.sum(RNA_2d_fea * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)
        m = m0*r0  # hidden
        for DMA_iter in range(self.DMA_depth):
            mol_to_rna = torch.matmul(pairwise_pred_2d, F.tanh(self.m_to_r_transform[DMA_iter](Mol_2d_fea)))  # batch * n_base * hidden
            rna_to_mol = torch.matmul(pairwise_pred_2d.transpose(1,2), F.tanh(self.r_to_m_transform[DMA_iter](RNA_2d_fea)))  # batch * n_vertex * hidden
            mol_tmp = F.tanh(self.hc0[DMA_iter](Mol_2d_fea))*F.tanh(self.mc1[DMA_iter](m)).view(batch_size,1,-1)*rna_to_mol #batch * n_vertex *hidden
            rna_tmp = F.tanh(self.hp0[DMA_iter](RNA_2d_fea))*F.tanh(self.mp1[DMA_iter](m)).view(batch_size,1,-1)*mol_to_rna #batch * n_base *hidden
            mol_att = self.mask_softmax(self.hc1[DMA_iter](mol_tmp).view(batch_size,-1), vertex_mask.view(batch_size,-1)) 
            rna_att = self.mask_softmax(self.hp1[DMA_iter](rna_tmp).view(batch_size,-1), seq_mask.view(batch_size,-1))
            mf = torch.sum(Mol_2d_fea*mol_att.view(batch_size,-1,1), dim=1)  #mol_att is processed by softmax
            rf = torch.sum(RNA_2d_fea*rna_att.view(batch_size,-1,1), dim=1)
            m = self.GRU_dma(m, mf*rf)
        return mf, rf
    
    def dma_gru_3d(self, batch_size, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_3d ):
        vertex_mask = Mol_mask.unsqueeze(-1)#.view(batch_size,-1,1)
        seq_mask = RNA_mask.unsqueeze(-1)#.view(batch_size,-1,1)
        m0 = torch.sum(Mol_feats * vertex_mask, dim=1) / torch.sum(vertex_mask, dim=1)
        r0 = torch.sum(RNA_feats * seq_mask, dim=1) / torch.sum(seq_mask, dim=1)
        m = m0 * r0
        for DMA_iter in range(self.DMA_depth):
            mol_to_rna = torch.matmul(pairwise_pred_3d, F.tanh(self.m_to_r_transform_3d[DMA_iter](Mol_feats)))  # batch * n_residue * hidden
            rna_to_mol = torch.matmul(pairwise_pred_3d.transpose(1,2), F.tanh(self.r_to_m_transform_3d[DMA_iter](RNA_feats)))  # batch * n_vertex * hidden
            
            mol_tmp = F.tanh(self.hc0_3d[DMA_iter](Mol_feats))*F.tanh(self.mc1_3d[DMA_iter](m)).unsqueeze(-2)*rna_to_mol
            rna_tmp = F.tanh(self.hp0_3d[DMA_iter](RNA_feats))*F.tanh(self.mp1_3d[DMA_iter](m)).unsqueeze(-2)*mol_to_rna
            mol_att = self.mask_softmax(self.hc1_3d[DMA_iter](mol_tmp).squeeze(-1), vertex_mask.squeeze(-1)) 
            rna_att = self.mask_softmax(self.hp1_3d[DMA_iter](rna_tmp).squeeze(-1), seq_mask.squeeze(-1))

            mf = torch.sum(Mol_feats*mol_att.unsqueeze(-1), dim=1)
            rf = torch.sum(RNA_feats*rna_att.unsqueeze(-1), dim=1)
            m = self.GRU_dma_3d(m, mf*rf)
        return mf, rf

    def forward(self, RNA_repre = None, Seq_mask = None, RNA_Graph = None, Mol_Graph = None, RNA_feats = None, RNA_C4_coors = None, RNA_coors = None, RNA_mask = None, Mol_feats = None, Mol_coors = None, Mol_mask = None, Mol_LAS_dis=None): #改动，增加Las_dis，即几何距离约束
        if self.rna_graph:
            RNA_2d_fea = self.GCN_rna( RNA_Graph )
            RNA_2d_fea, seq_mask = to_dense_batch(RNA_2d_fea, RNA_Graph.batch)
            seq_mask = seq_mask.float().to(RNA_2d_fea.device)#cuda(5)
        if self.mol_graph:
            Mol_2d_fea = self.GCN_mol( Mol_Graph )
            Mol_2d_fea, vertex_mask = to_dense_batch(Mol_2d_fea, Mol_Graph.batch)
            vertex_mask = vertex_mask.float().to(Mol_2d_fea.device)
        if self.rna_repre:
            RNA_1d_fea = self.mlp_rna( RNA_repre )
            seq_mask = Seq_mask.float().to(RNA_1d_fea.device)
        if self.coors:
            Mol_feats = self.equi_mol(Mol_feats, Mol_coors, Mol_mask).type0  # batch * max(length) * 16 (mini_size for equibind)
            RNA_feats = self.equi_rna(RNA_feats, RNA_coors, RNA_mask).type0

            RNA_mask = RNA_mask.float()
            Mol_mask = Mol_mask.float()
        #print(RNA_1d_fea.shape, RNA_feats.shape, Mol_feats.shape )
        assert self.mol_graph or self.coors
        if self.mol_graph:
            batch_size = seq_mask.size(0)
        elif self.coors:
            batch_size = RNA_feats.size(0)
                                   
        if self.mol_graph:
            if self.coors:
                if self.rna_graph:
                    if self.rna_repre:
                        pairwise_pred_2d, pairwise_pred_3d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
                    else:
                        pairwise_pred_2d, pairwise_pred_3d = self.Pairwise_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
                else:
                    pairwise_pred_2d, pairwise_pred_3d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
            else:
                if self.rna_graph:
                    if self.rna_repre:
                        pairwise_pred_2d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, RNA_C4_coors, None, Mol_coors, None, None, Mol_LAS_dis)
                    else:
                        pairwise_pred_2d = self.Pairwise_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, RNA_C4_coors, None, Mol_coors, None, None, Mol_LAS_dis)
                else: 
                    pairwise_pred_2d = self.Pairwise_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, None, None, RNA_C4_coors, None, Mol_coors, None, None, Mol_LAS_dis)
        else:
            pairwise_pred_3d = self.Pairwise_pred_module( batch_size, None, None, None, None, None, RNA_feats, Mol_feats, RNA_C4_coors, RNA_coors, Mol_coors, RNA_mask, Mol_mask, Mol_LAS_dis)
        
        if self.mol_graph:
            if self.coors:
                if self.rna_repre:
                    if self.rna_graph:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_2d, pairwise_pred_3d )
                    else:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_2d, pairwise_pred_3d )
                else:
                    affinity_pred = self.Affinity_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, RNA_feats, Mol_feats, RNA_mask, Mol_mask, pairwise_pred_2d, pairwise_pred_3d )
            else:
                if self.rna_repre:
                    if self.rna_graph:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, None, None, pairwise_pred_2d, None )
                    else:
                        affinity_pred = self.Affinity_pred_module( batch_size, RNA_1d_fea, None, Mol_2d_fea, seq_mask, vertex_mask, None, None, None, None, pairwise_pred_2d, None )
                else:
                    affinity_pred = self.Affinity_pred_module( batch_size, None, RNA_2d_fea, Mol_2d_fea, seq_mask, vertex_mask, None, None, None, None, pairwise_pred_2d, None )
        else:
            affinity_pred = self.Affinity_pred_module( batch_size, None, None, None, None, None, RNA_feats, Mol_feats, RNA_mask, Mol_mask, None, pairwise_pred_3d )
            return affinity_pred, pairwise_pred_3d

        return affinity_pred, pairwise_pred_2d