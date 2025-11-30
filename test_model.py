import pickle
from torch.utils import data
from torch_geometric.data import Batch
import torch.utils.data.sampler as sampler
import numpy as np
import sys
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data
from torch.autograd import Variable
import torch.optim as optim
import random
from utils.net_utils import *
from utils.metrics import *
from net.model import GerNA
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from edl_pytorch import Dirichlet, evidential_classification,evidential_regression
from tqdm import tqdm
import argparse
from data_utils.dataset import GerNA_dataset, custom_collate_fn
import json
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.multiprocessing as mp

def set_random_seeds(seed_value=42):
    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seeds(seed_value=99)

def test(net, dataLoader, batch_size, mode, device, threshold = 0, uncertainty_mode = True):
    output_list = []
    label_list = []
    pairwise_auc_list = []
    confidence_list = []
    mu_list, v_list, alpha_list, beta_list = [],[],[],[]
    with torch.no_grad():
        net.eval()
        for batch_index, [batch_RNA_repre, batch_seq_mask, batch_Mol_Graph, batch_RNA_Graph, batch_RNA_feats, batch_RNA_C4_coors,batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS, batch_label] in enumerate(dataLoader):
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
            batch_label = batch_label.to(device)
            affinity_label = batch_label
            affinity_pred, _  = net( batch_RNA_repre, batch_seq_mask, batch_RNA_Graph, batch_Mol_Graph, batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS )
            output_list += affinity_pred.cpu().detach().numpy().tolist()
            label_list += affinity_label.reshape(-1).tolist()
        output_list = np.array(output_list)
        label_list = np.array(label_list)
        probs = []
        uncertainty = []
        for alpha in output_list:
            probs.append(alpha[1] / alpha.sum())
        new_output_list = np.array(probs)
    
    if mode == "train":
        mcc_threshold, TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, max_mcc, AUC, AUPRC = get_train_metrics( new_output_list.reshape(-1),label_list.reshape(-1))
        test_performance = [ mcc_threshold, TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, max_mcc, AUC, AUPRC ]
        return test_performance, label_list, output_list
    elif mode == "valid":
        TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, mcc, AUC, AUPRC = get_valid_metrics(new_output_list.reshape(-1),label_list.reshape(-1),threshold )
        test_performance = [TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, mcc, AUC, AUPRC ]
        return test_performance, label_list, output_list
    elif mode == "test":
        TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, mcc, AUC, AUPRC = get_valid_metrics(new_output_list.reshape(-1),label_list.reshape(-1),threshold )
        test_performance = [TN, FN, FP, TP, Pre, Sen, Spe, Acc, F1_score, mcc, AUC, AUPRC ]
        return test_performance, label_list, output_list

def eval(rank, world_size, trainDataset, trainUnbDataset, validDataset, testDataset, params, batch_size=8, num_epoch=30, model_path=None):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12348'
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=60))
    
    train_sampler = DistributedSampler(trainDataset, num_replicas=world_size,
                                       rank=rank)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size,
                              sampler=train_sampler,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True,drop_last=True)
    if rank==0:
        train_unb_DataLoader = torch.utils.data.DataLoader(trainUnbDataset, batch_size=batch_size,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True)
        validDataLoader = torch.utils.data.DataLoader(validDataset, batch_size=batch_size,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True)
        testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True)
    
    net = GerNA(params, trigonometry = True, rna_graph = True, coors = True, coors_3_bead = True, uncertainty=True)  #define the network
    
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path,map_location="cuda:{}".format(rank))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
        print("Load successfully!")
    else:
        net.apply(weights_init)

    threshold = 0
    net = net.to(device)

    net = DistributedDataParallel(net, device_ids=[rank])
    net._set_static_graph()
    
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    max_auroc = 0
    
    train_loss = []

    train_output_list = []
    train_label_list = []
    total_loss = 0
    affinity_loss = 0
    conf_loss= 0
    pairwise_loss = 0
    
    if rank==0:
        perf_name = ['TN', 'FN', 'FP', 'TP', 'Pre', 'Sen', 'Spe', 'Acc', 'F1_score', 'Mcc', 'AUC', 'AUPRC']
        train_performance, train_label, train_output = test(net.module, train_unb_DataLoader, batch_size, "train",device, uncertainty_mode=True)
        threshold = train_performance[0]
        print('threshold:',threshold )
        print_perf = [perf_name[i]+' '+str(round(train_performance[i+1], 6)) for i in range(len(perf_name))]
        print( 'train', len(train_output), ' '.join(print_perf))

        test_performance, test_label, test_output = test(net.module, testDataLoader, batch_size,"test", device, threshold,uncertainty_mode = True)
        print_perf = [perf_name[i]+' '+str(round(test_performance[i], 6)) for i in range(len(perf_name))]
        print('test ', len(test_output), ' '.join(print_perf))
    dist.barrier()
        
    if rank==0:
        print('Finished Training')

    dist.barrier()
    dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    parser.add_argument('--dataset', type=str, default='Robin', choices=['Robin', 'Biosensor'], help='Path to the dataset file')
    parser.add_argument('--split_method', type=str, default='random', choices=['random', 'RNA', 'mol', 'both'], help='Method to split the dataset')
    parser.add_argument('--model_path', type=str, default='Model/Robin_Model_baseline.pth', help='Path to load the model')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')

    parser.add_argument('--GNN_depth', type=int, default=4, help='Depth of the GNN')
    parser.add_argument('--DMA_depth', type=int, default=2, help='Depth of the DMA')
    parser.add_argument('--hidden_size1', type=int, default=128, help='Size of the first hidden layer')
    parser.add_argument('--hidden_size2', type=int, default=128, help='Size of the second hidden layer')
    parser.add_argument('--cuda', type=str, default="0", help='Device to use, e.g., "cuda:0", "cuda:1" ')

    args = parser.parse_args()

    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    dataset = args.dataset
    split_method = args.split_method


    train_path = "/data/ypxia/github/GerNA-Bind/open_data/"+dataset+"/"+split_method+"/train_data.pkl"
    valid_path = "/data/ypxia/github/GerNA-Bind/open_data/"+dataset+"/"+split_method+"/valid_data.pkl"
    test_path = "/data/ypxia/github/GerNA-Bind/open_data/"+dataset+"/"+split_method+"/test_data.pkl"
    trainDataset = GerNA_dataset(train_path)
    validDataset = GerNA_dataset(valid_path)
    testDataset = GerNA_dataset(test_path)

    n_epoch = args.epoch
    batch_size = args.batch_size
    params = [args.GNN_depth, args.DMA_depth, args.hidden_size1, args.hidden_size2]
    model_path = args.model_path
    
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    func_args = (world_size, trainDataset, trainDataset, validDataset,testDataset,params, batch_size, n_epoch, model_path)
    mp.spawn(eval, args=func_args, nprocs=world_size, join=True)
