import numpy as np
import torch
import scipy
import random
import torch.nn as nn
import math


def get_mask(arr_list):
    N = max(len(x) if isinstance(x, list) else x.shape[0] for x in arr_list)
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        if isinstance(arr, list):
            n = len(arr)
        else:
            n = arr.shape[0]
        a[i,:n] = 1
    return a

def get_mask_RNA_one(arr_list):
    #N = max(len(x) if isinstance(x, list) else x.shape[0] for x in arr_list)
    N=100
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        if isinstance(arr, list):
            n = len(arr)
        else:
            n = arr.shape[0]
        a[i,:n] = 1
    return a

def pack1D(arr_list):
    N = max(len(x) if isinstance(x, list) else x.shape[0] for x in arr_list)
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        if isinstance(arr, list):
            n = len(arr)
        else:
            n = arr.shape[0]
        a[i,0:n] = arr
    return a

def pack2D_RNA_one(arr_list):  
    N = 100
    M = max(len(x[0]) if isinstance(x, list) else x.shape[1] for x in arr_list)
    a = np.zeros((len(arr_list), N, M))
    
    for i, arr in enumerate(arr_list):
        if isinstance(arr, list):
            n = min(N, len(arr))
            m = len(arr[0])
            temp_arr = np.array(arr[:n])
            if n < N:
                temp_arr = np.concatenate((temp_arr, np.zeros((N - n, m))))
        else:
            n = min(N, arr.shape[0])
            m = arr.shape[1]
            temp_arr = arr[:n, :M]
            if n < N or m < M:
                temp_arr = np.pad(temp_arr, ((0, N - n), (0, M - m)), 'constant')
        a[i, :, :] = temp_arr[:N, :M]
    return a

def pack2D(arr_list):
    N = max(len(x) if isinstance(x, list) else x.shape[0] for x in arr_list)
    M = max(len(x[0]) if isinstance(x, list) else x.shape[1] for x in arr_list)
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        if isinstance(arr, list):
            n = len(arr)
            m = len(arr[0])
        else:
            n = arr.shape[0]
            m = arr.shape[1]
        a[i,0:n,0:m] = arr  
    return a

def count_ones(arr):
    counts = []
    for batch in arr:
        count = -1
        zero_added = False
        result = []
        for num in batch:
            if num == 0 and not zero_added:
                count += 1
                zero_added = True
            elif num == 1 and not zero_added:
                count += 1
            result.append(count)
        counts.append(result)
    return torch.tensor(counts)

def expand_matrix(indices, matrix):
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    expanded_matrices = []
    for batch_indices, batch_matrix in zip(indices, matrix):
        batch_indices = batch_indices.to(device)
        batch_matrix = batch_matrix.to(device)

        n = len(batch_indices)
        expanded_matrix = torch.zeros(n, n, dtype=batch_matrix.dtype, device=device)

        for i, idx_i in enumerate(batch_indices):
            for j, idx_j in enumerate(batch_indices):
                expanded_matrix[i][j] = batch_matrix[idx_i][idx_j]
        expanded_matrices.append(expanded_matrix)
    return torch.stack(expanded_matrices)


def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    return pair_dis_one_hot

#This is for more molecule.
def get_mol_pair_dis_distribution(coords, LAS_distance_constraint_mask=None): #convert coords to one-hot distance map
    #pair_dis = scipy.spatial.distance.cdist(coords, coords)
    pair_dis = torch.cdist(coords, coords, p=2)
    #print(pair_dis)
    bin_size=1
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        # for i in range(pair_dis.shape[1]):
        #     pair_dis[i, i] = 0
    #pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis = pair_dis.clone().detach()
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    
    return pair_dis_distribution

def sum_tensor_by_order(tensor_array, indices):
    result_tensor = torch.zeros(torch.max(indices), tensor_array.size(1))

    for i, idx in enumerate(indices):
        result_tensor[idx - 1] += tensor_array[i]

    return result_tensor

def generate_ratios(indices):
    index_positions = {}
    ratios = []
    for idx in indices:
        if idx in index_positions:
            index_positions[idx] += 1
        else:
            index_positions[idx] = 1
        ratio = 1 / index_positions[idx]
        ratios.append(ratio)
    return [1 / count for count in index_positions.values()]


def scale_matrix_by_ratios(matrix, ratios):
    start_idx = 0

    for ratio in ratios:
        end_idx = start_idx + 1#int(matrix.size(1) * ratio)
        matrix[start_idx:end_idx,:] = matrix[start_idx:end_idx,:] * ratio
        start_idx = end_idx
    return matrix

def random_with_probability(prob_true):
    """random return bool"""
    rand_num = random.random()
    return rand_num < prob_true

def weights_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m,nn.Linear):
        nn.init.normal_(m.weight.data, mean=0, std=min( 1/ math.sqrt(m.weight.data.shape[-1]), 0.03))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)