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
#sys.path.append("net/")
from net.model import GerNA
#torch.ops.torch_use_cuda_dsa(True)
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
    # random seed in python
    random.seed(seed_value)
    
    # Numpy
    np.random.seed(seed_value)
    
    # PyTorch
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
        #regression
        #combined_affinity_pred = [mu_list,v_list,alpha_list,beta_list]
    
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
    
#train-eval-test function
def train_and_eval(rank,world_size, trainDataset, trainUnbDataset, validDataset, testDataset, params, batch_size=8, num_epoch=30, model_path=None):
    # print("sss")
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=60))
    
    train_sampler = DistributedSampler(trainDataset, num_replicas=world_size,
                                       rank=rank)
    trainDataLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size,
                              sampler=train_sampler,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True,drop_last=True)
    if rank==0:
        train_unb_DataLoader = torch.utils.data.DataLoader(trainUnbDataset, batch_size=batch_size,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True)
        validDataLoader = torch.utils.data.DataLoader(validDataset, batch_size=batch_size,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True)
        testDataLoader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size,collate_fn=custom_collate_fn,num_workers=10,pin_memory=True)
        if not os.path.exists(model_path):
            os.makedirs(model_path)  #this should need to run in rank 0 also, if not, it will conflict in multi threads.
    ak_model_path = model_path
    model_path = model_path + "Model_baseline.pth"
    
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

    #net = torch_geometric.nn.DataParallel(net.cuda())
    threshold = 0
    net = net.to(device)

    net = DistributedDataParallel(net, device_ids=[rank])
    net._set_static_graph()
    
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('total num params', pytorch_total_params)

    criterion1 = nn.MSELoss()
    soft_loss = nn.CrossEntropyLoss()
    confidence_criterion = nn.BCELoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0003, weight_decay=0, amsgrad=True)  #0.0005
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    max_auroc = 0
    
    train_loss = []

    for epoch in range(num_epoch):
        
        for param_group in optimizer.param_groups:
            print('learning rate:', param_group['lr'])
        train_output_list = []
        train_label_list = []
        total_loss = 0
        affinity_loss = 0
        conf_loss= 0
        pairwise_loss = 0
        net.train()
        
        total_loss = torch.zeros(2).to(rank)
        
        for batch_index, [batch_RNA_repre, batch_seq_mask, batch_Mol_Graph, batch_RNA_Graph, batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS, batch_label] in enumerate(trainDataLoader):
            if batch_index % 1000 == 0:
               print('epoch', epoch, 'batch', batch_index)
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
            affinity_label = batch_label.reshape(-1,1)
            
            optimizer.zero_grad()
            
            affinity_pred, _ = net( batch_RNA_repre, batch_seq_mask, batch_RNA_Graph, batch_Mol_Graph, batch_RNA_feats, batch_RNA_C4_coors, batch_RNA_coors, batch_RNA_mask, batch_Mol_feats, batch_Mol_coors, batch_Mol_mask, batch_Mol_LAS )
            loss = evidential_classification(  affinity_pred,  affinity_label.long().reshape(-1), lamb=0.1 )
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            total_loss[0] += float(loss.data*len(batch_RNA_repre))
            total_loss[1] += len(batch_RNA_repre)
            
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        loss = float(total_loss[0] / total_loss[1])
        scheduler.step()
        train_loss.append( total_loss )

        if rank==0:
            perf_name = ['TN', 'FN', 'FP', 'TP', 'Pre', 'Sen', 'Spe', 'Acc', 'F1_score', 'Mcc', 'AUC', 'AUPRC']
            train_performance, train_label, train_output = test(net.module, train_unb_DataLoader, batch_size, "train",device, uncertainty_mode=True)

            threshold = train_performance[0]
            print('threshold:',threshold )
            print_perf = [perf_name[i]+' '+str(round(train_performance[i+1], 6)) for i in range(len(perf_name))] #第一个是threshold
            print( 'train', len(train_output), ' '.join(print_perf))

            valid_performance, valid_label, valid_output = test(net.module, validDataLoader, batch_size, "valid", device, threshold, uncertainty_mode = True)
            print_perf = [perf_name[i]+' '+str(round(valid_performance[i], 6)) for i in range(len(perf_name))]
            print('valid', len(valid_output), ' '.join(print_perf) )

            if valid_performance[-2] > max_auroc:
                max_auroc = valid_performance[-2]
                #revise here for DDP train
                #torch.save(net.state_dict(), model_path)
                torch.save(net.module.state_dict(), model_path)
                test_performance, test_label, test_output = test(net.module, testDataLoader, batch_size,"test", device, threshold,uncertainty_mode = True)
                print_perf = [perf_name[i]+' '+str(round(test_performance[i], 6)) for i in range(len(perf_name))]
                print('test ', len(test_output), ' '.join(print_perf))
        dist.barrier()
        
    if rank==0:
        print('Finished Training')
        # data = [test_performance, test_label, test_output, train_loss, train_A, train_B, train_C, valid_A, valid_B, valid_C, test_A, test_B, test_C]
        # with open(ak_model_path+"baseline_2.pickle", 'wb') as f:
        #     pickle.dump(data,f)

    dist.barrier()
    dist.destroy_process_group()
        #return test_performance, test_label, test_output, train_loss, train_A, train_B, train_C, valid_A, valid_B, valid_C, test_A, test_B, test_C

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train and evaluate the model')
    parser.add_argument('--dataset', type=str, default='Robin', help='Path to the dataset file')
    parser.add_argument('--split_method', type=str, default='KFold', choices=['random', 'RNA', 'mol', 'both'], help='Method to split the dataset')
    parser.add_argument('--model_output_path', type=str, default='Model/', help='Path to save the model')
    parser.add_argument('--epoch', type=int, default=12, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size for training')

    parser.add_argument('--GNN_depth', type=int, default=4, help='Depth of the GNN')
    parser.add_argument('--DMA_depth', type=int, default=2, help='Depth of the DMA')
    parser.add_argument('--hidden_size1', type=int, default=128, help='Size of the first hidden layer')
    parser.add_argument('--hidden_size2', type=int, default=128, help='Size of the second hidden layer')
    parser.add_argument('--cuda', type=str, default=0, help='Device to use, e.g., "cuda:0", "cuda:1" ')

    args = parser.parse_args()

    if args.dataset == 'Robin':
        dataset_path = "data/Robin/Robin_all_data_3_coors_C4.pkl"
    elif args.dataset == 'Biosensor':
        dataset_path = "data/Biosensor/Biosensor_all_data_3_coors_C4.pkl"

    my_Dataset = GerNA_dataset(dataset_path)
    all_data_index = [ i for i in range(len(my_Dataset)) ]

    with open(dataset_path,'rb') as f:
        _,_,_,_,_,_,_,_,_,label = pickle.load(f)

    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda


    if args.split_method == 'random':
        kfold = KFold(n_splits=5, shuffle=True)
        train_index, test_index = next(kfold.split(all_data_index))
        print("Selected fold")
        eval_index = train_index[:int(0.2 * len(train_index))]
        train_index = train_index[int(0.2 * len(train_index)):]
#        train_index = train_index[int(0.99 * len(train_index)):]
    elif args.split_method == 'RNA':
        with open("data/"+args.dataset+"/"+args.dataset+"_"+args.split_method+".json", "r") as json_file:
            json_data = json.load(json_file)
            train_index = json_data['train']
            eval_index = json_data['eval']
            test_index = json_data['test']
    elif args.split_method == 'mol':
        with open("data/"+args.dataset+"/"+args.dataset+"_"+args.split_method+".json", "r") as json_file:
            json_data = json.load(json_file)
            train_index = json_data['train']
            eval_index = json_data['eval']
            test_index = json_data['test']
    elif args.split_method == 'both':
        with open("data/"+args.dataset+"/"+args.dataset+"_RNA.json", "r") as json_file:
            json_data_RNA = json.load(json_file)
            train_index_rna = json_data_RNA['train']
            eval_index_rna = json_data_RNA['eval']
            test_index_rna = json_data_RNA['test']

        with open("data/"+args.dataset+"/"+args.dataset+"_mol.json", "r") as json_file:
            json_data_mol = json.load(json_file)
            train_index_mol = json_data_mol['train']
            eval_index_mol = json_data_mol['eval']
            test_index_mol = json_data_mol['test']

            train_index = list( set(train_index_rna).intersection(set(train_index_mol)) )
            eval_index = list( set(eval_index_rna).intersection(set(eval_index_mol)) )
            test_index = list( set(test_index_rna).intersection(set(test_index_mol)) )

    train_index_bal = list(train_index[:])
    for i in range(len(train_index)):
        if label[train_index_bal[i]]==1:
            train_index_bal.extend( [train_index_bal[i]]*10 )

    trainDataset = data.Subset(my_Dataset,train_index_bal)
    trainUnbDataset = data.Subset(my_Dataset,train_index)
    validDataset = data.Subset(my_Dataset,eval_index)
    testDataset = data.Subset(my_Dataset,test_index)

    n_epoch = args.epoch
    batch_size = args.batch_size
    params = [args.GNN_depth, args.DMA_depth, args.hidden_size1, args.hidden_size2]
    model_path = args.model_output_path
    
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    func_args = (world_size, trainDataset, trainUnbDataset, validDataset,testDataset,params, batch_size, n_epoch, model_path)
    mp.spawn(train_and_eval, args=func_args, nprocs=world_size, join=True)
