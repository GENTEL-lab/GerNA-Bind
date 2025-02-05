from Bio.PDB import PDBParser
import pandas as pd
import numpy as np
import os
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
import torch.nn.functional as F
from io import StringIO
import sys
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select
import scipy
import scipy.spatial
import requests
from rdkit.Geometry import Point3D

from torchdrug import data as td     # conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg if fail to import

'''
Revised by Yunpeng Xia
read_mol function: get the molecule conformation by sdf, then mol2 file.
write_renumbered_sdf function: let the atoms in molecules renumbered.
get_clean_res_list function: sift the atoms in the residue.
get_compound_pair_dis_distribution function: get the coordinates into one-hot encoding map.
extract_torchdrug_feature_from_mol function: use torch_drug data to generate the molecule graph data.
select_chain_within_cutoff_to_ligand_v2 function: get the distance < 10 residue partner into the file.
get_protein_feature function: use gvp.data.ProteinGraphDataset and (seq,coords) to generate the protein graph data
'''

def read_mol(sdf_fileName, mol2_fileName, verbose=False):
    Chem.WrapLogs()
    stderr = sys.stderr
    sio = sys.stderr = StringIO()
    mol = Chem.MolFromMolFile(sdf_fileName, sanitize=False)
    problem = False
    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        sm = Chem.MolToSmiles(mol)
    except Exception as e:
        sm = str(e)
        problem = True
    if problem:
        mol = Chem.MolFromMol2File(mol2_fileName, sanitize=False)
        problem = False
        try:
            Chem.SanitizeMol(mol)
            mol = Chem.RemoveHs(mol)
            sm = Chem.MolToSmiles(mol)
            problem = False
        except Exception as e:
            sm = str(e)
            problem = True

    if verbose:
        print(sio.getvalue())
    sys.stderr = stderr
    return mol, problem


def write_renumbered_sdf(toFile, sdf_fileName, mol2_fileName):
    # read in mol
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    # reorder the mol atom number as in smiles.
    m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)['_smilesAtomOutputOrder'])
    mol = Chem.RenumberAtoms(mol, m_order)
    w = Chem.SDWriter(toFile)
    w.write(mol)
    w.close()

def get_canonical_smiles(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    mol = Chem.MolFromSmiles(smiles)
    # mol = Chem.RemoveAllHs(mol)
    # mol = Chem.AddHs(mol)
    ps = AllChem.ETKDGv2()
    # rid = AllChem.EmbedMolecule(mol, ps)
    for repeat in range(n_repeat):
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == 0:
            break
    if rid == -1:
        print("rid", pdb, rid)
        ps.useRandomCoords = True
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == -1:
            mol.Compute2DCoords()
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    # mol = Chem.RemoveAllHs(mol)
    return mol


def binarize(x):
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))

#adj - > n_hops connections adj
def n_hops_adj(adj, n_hops):
    adj_mats = [torch.eye(adj.size(0), dtype=torch.long, device=adj.device), binarize(adj + torch.eye(adj.size(0), dtype=torch.long, device=adj.device))]

    for i in range(2, n_hops+1):
        adj_mats.append(binarize(adj_mats[i-1] @ adj_mats[1]))
    extend_mat = torch.zeros_like(adj)

    for i in range(1, n_hops+1):
        extend_mat += (adj_mats[i] - adj_mats[i-1]) * i

    return extend_mat

def get_LAS_distance_constraint_mask(mol):
    # Get the adj
    adj = Chem.GetAdjacencyMatrix(mol)
    adj = torch.from_numpy(adj)
    extend_adj = n_hops_adj(adj,2)
    # add ring
    ssr = Chem.GetSymmSSSR(mol)
    for ring in ssr:
        # print(ring)
        for i in ring:
            for j in ring:
                if i==j:
                    continue
                else:
                    extend_adj[i][j]+=1
    # turn to mask
    mol_mask = binarize(extend_adj)
    return mol_mask

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=None): #convert coords to one-hot distance map
    pair_dis = scipy.spatial.distance.cdist(coords, coords)
    bin_size=1
    bin_min=-0.5
    bin_max=15
    if LAS_distance_constraint_mask is not None:
        pair_dis[LAS_distance_constraint_mask==0] = bin_max
        # diagonal is zero.
        for i in range(pair_dis.shape[0]):
            pair_dis[i, i] = 0
    pair_dis = torch.tensor(pair_dis, dtype=torch.float)
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=16)
    pair_dis_distribution = pair_dis_one_hot.float()
    return pair_dis_distribution


def extract_torchdrug_feature_from_mol(mol, has_LAS_mask=False):  #torcdrug_data generate the mol graph data
    coords = mol.GetConformer().GetPositions()
    if has_LAS_mask:
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol)
    else:
        LAS_distance_constraint_mask = None
    pair_dis_distribution = get_compound_pair_dis_distribution(coords, LAS_distance_constraint_mask=LAS_distance_constraint_mask)
    molstd = td.Molecule.from_smiles(Chem.MolToSmiles(mol),node_feature='property_prediction') #torcdrug_data
    # molstd = td.Molecule.from_molecule(mol ,node_feature=['property_prediction'])
    compound_node_features = molstd.node_feature # nodes_chemical_features
    edge_list = molstd.edge_list # [num_edge, 3]
    edge_weight = molstd.edge_weight # [num_edge, 1]
    assert edge_weight.max() == 1
    assert edge_weight.min() == 1
    assert coords.shape[0] == compound_node_features.shape[0]
    edge_feature = molstd.edge_feature # [num_edge, edge_feature_dim]
    x = (coords, compound_node_features, edge_list, edge_feature, pair_dis_distribution)
    return x