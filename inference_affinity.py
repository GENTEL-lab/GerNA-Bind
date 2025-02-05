import pickle
from torch.utils import data
from torch_geometric.data import Batch
import torch.utils.data.sampler as sampler
import numpy as np
import pandas as pd

import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.optim as optim
import random
from utils.net_utils import *
from utils.metrics import *
#sys.path.append("net/")

from net.model import GerNA
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from edl_pytorch import Dirichlet, evidential_classification,evidential_regression
from tqdm import tqdm

class Test_dataset(data.Dataset):
    def __init__(self, path):
        with open(path,'rb') as f:
            self.data = pickle.load(f)
        self.RNA_repre,self.Mol_graph,self.RNA_Graph,self.RNA_feats,self.RNA_C4_coors,self.RNA_coors,self.Mol_feats,self.Mol_coors,self.LAS_input = self.data
        
    def __len__(self):
        return len(self.RNA_repre)

    def __getitem__(self, index):
        return self.RNA_repre[index],self.Mol_graph[index],self.RNA_Graph[index],self.RNA_feats[index],self.RNA_C4_coors[index],self.RNA_coors[index],self.Mol_feats[index],self.Mol_coors[index],self.LAS_input[index]
    
def custom_collate_fn(batch):
    batch_RNA_repre = []
    batch_seq_mask = []
    batch_Mol_Graph = []
    batch_RNA_Graph = []
    batch_RNA_feats = []
    batch_RNA_C4_coors = []
    batch_RNA_coors = []
    batch_RNA_mask = []
    batch_Mol_feats = []
    batch_Mol_coors = []
    batch_Mol_mask = []
    batch_Mol_LAS = []

    for i,item in enumerate(batch):
        rna_repre,mol_graph,rna_graph,rna_feats,rna_coors_C4,rna_coors,mol_feats,mol_coors,mol_las = item
        
        batch_RNA_repre.append(rna_repre)
        batch_Mol_Graph.append(mol_graph)
        batch_RNA_Graph.append(rna_graph)
        
        batch_RNA_feats.append(rna_feats)
        batch_RNA_C4_coors.append(rna_coors_C4)
        batch_RNA_coors.append(rna_coors)

        batch_Mol_feats.append(mol_feats)
        batch_Mol_coors.append(mol_coors)

        batch_Mol_LAS.append(mol_las)
            
    batch_seq_mask = torch.tensor( np.array(get_mask(batch_RNA_repre))).float()#.to(device)
    batch_RNA_repre = torch.tensor( np.array(pack2D(batch_RNA_repre))).float()#.to(device)
    
    batch_Mol_Graph = Batch.from_data_list(batch_Mol_Graph)#.to(device)
    batch_RNA_Graph = Batch.from_data_list(batch_RNA_Graph)#.to(device)
    
    batch_RNA_mask = torch.tensor( np.array(get_mask(batch_RNA_feats))).bool()#.to(device)
    batch_Mol_mask = torch.tensor( np.array(get_mask(batch_Mol_coors))).bool()#.to(device)
    
    batch_RNA_feats = torch.tensor( np.array(pack1D(batch_RNA_feats))).long()#.to(device)
    batch_RNA_C4_coors = torch.tensor( np.array(pack2D(batch_RNA_C4_coors)) ).float()
    batch_RNA_coors = torch.tensor( np.array(pack2D(batch_RNA_coors))).float()#.to(device)
    
    batch_Mol_feats = torch.tensor( np.array(pack1D(batch_Mol_feats))).long()#.to(device)
    batch_Mol_coors = torch.tensor( np.array(pack2D(batch_Mol_coors))).float()#.to(device)
    
    batch_Mol_LAS = torch.tensor( np.array(pack2D(batch_Mol_LAS))).float()#.to(device)

    return batch_RNA_repre, batch_seq_mask, batch_Mol_Graph, batch_RNA_Graph, batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_data = Test_dataset("data/new_data.pkl")

all_data_index = [ i for i in range(len(test_data)) ]

eval_index = all_data_index
valid_samples = sampler.SequentialSampler(eval_index)

batch_size = 1
validDataLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, sampler=valid_samples, collate_fn=custom_collate_fn,num_workers=4,pin_memory=True)
    
params = [4, 2, 128, 128]
model_path = "Model/Robin_Model_baseline.pth"

net = GerNA(params, trigonometry = True, rna_graph = True, coors = True, coors_3_bead = True, uncertainty=True)

if os.path.exists(model_path):
    pretrained_dict = torch.load(model_path,map_location="cuda")
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
net = net.to(device)


output_list = []
#pairwise_list = []
confidence_list = []
new_output_list = []
mu_list, v_list, alpha_list, beta_list = [],[],[],[]

with torch.no_grad():
    net.eval()
    for batch_index, [batch_RNA_repre, batch_seq_mask, batch_Mol_Graph, batch_RNA_Graph, batch_RNA_feats, batch_RNA_C4_coors,batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS] in enumerate(validDataLoader):

        batch_RNA_repre = batch_RNA_repre.to(device)
        batch_seq_mask = batch_seq_mask.to(device)
        batch_Mol_Graph = batch_Mol_Graph.to(device)
        batch_RNA_Graph = batch_RNA_Graph.to(device)
        batch_RNA_feats = batch_RNA_feats.to(device)
        batch_RNA_C4_coors = batch_RNA_C4_coors.to(device)
        batch_RNA_coors = batch_RNA_coors.to(device)
        batch_RNA_mask = batch_RNA_mask.to(device)
        batch_Mol_feats = batch_Mol_feats.to(device)
        batch_Mol_coors = batch_Mol_coors.to(device)
        batch_Mol_mask = batch_Mol_mask.to(device)
        batch_Mol_LAS = batch_Mol_LAS.to(device)

        affinity_pred,_  = net( batch_RNA_repre, batch_seq_mask, batch_RNA_Graph, batch_Mol_Graph, batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS )
        output_list += affinity_pred.cpu().detach().numpy().tolist()
        #pairwise_list += interaction

    output_list = np.array(output_list)
    probs = []
    uncertainty = []
    for alpha in output_list:
        probs.append(alpha[1] / alpha.sum())
        uncertainty.append( 2/alpha.sum() )
    new_output_list = np.array(probs)

row_names = [f"mol_{i+1}" for i in range(len(new_output_list))]
df = pd.DataFrame(new_output_list, index=row_names, columns=["Value"])
df.to_csv("affinity_prediction.csv")