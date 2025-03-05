import scanpy as sc
import anndata as ad

#import packages
import numpy as np
import json 
import scanpy as sc
from collections import OrderedDict
import scipy 
import pandas as pd
import matplotlib.pyplot as plt

#spectra imports 
import Spectra as spc
from Spectra import Spectra_util as spc_tl
from Spectra import K_est as kst
from Spectra import default_gene_sets


# read in /data/peer/sam/cytokine_central/data/processed/cui23/cui23_processed_spectrav2_example_20241216.h5ad
adata = sc.read_h5ad('/data/peer/sam/cytokine_central/data/processed/cui23/cui23_processed_spectrav2_example_20241216.h5ad')

# in adata.obs['cytokine_coarse'], rename 'IL-2.15' to 'IL-2-15'
adata.obs['cytokine_coarse'] = adata.obs['cytokine_coarse'].replace('IL-2.15', 'IL-2-15')

# read in /data/peer/sam/cytokine_central/references/gene_sets/spectra_dicts/spectra_genesetDict_Cuifull_TregcytokineSignaling_v2_20241107.json
import json
with open('/data/peer/sam/cytokine_central/references/gene_sets/spectra_dicts/spectra_genesetDict_Cuifull_TregcytokineSignaling_v2_20241107.json') as f:
    gene_dict = json.load(f)

# 
condition_specific_gene_sets = [
    "Treg_IL4",
    "Treg_TNFA",
    "Treg_TGFB1",
    "Treg_Th1",
    "Treg_IL2",
    "Treg_IFNG",
    "Treg_IFNA/B",
    "Treg_IL33",
    "hema_IL36",
    "Treg_Th17_UP",
    "hema_IL1B", 
    "lym_IL7",
    "lym_IL21",
    "lym_IL18"
]
gene_dict['PBS'] = {}
gene_dict['IFN-γ'] = {}
gene_dict['IL-2-15'] = {}

for key in condition_specific_gene_sets:
    gene_dict['PBS'][key] = gene_dict['global'][key]
    gene_dict['IFN-γ'][key] = gene_dict['global'][key]
    gene_dict['IL-2-15'][key] = gene_dict['global'][key]

# save the gene_dict as json
with open('/data/peer/rsaha/supervised-spectra/scripts/results/gene_dict.json', 'w') as f:
    json.dump(gene_dict, f)

# del gene_dict['IL-2.15']

# check problem
geneset_annotations_f = spc_tl.check_gene_set_dictionary(adata, gene_dict, obs_key = 'cytokine_coarse', global_key = 'global')

#filter gene set annotation dict for genes contained in adata
annotations = spc_tl.check_gene_set_dictionary(
    adata,
    gene_dict,
    obs_key='cytokine_coarse',
    global_key='global')

num_epochs = 10000
# fit the model (We will run this with only 2 epochs to decrease runtime in this tutorial)
model = spc.est_spectra(adata=adata, 
    gene_set_dictionary=annotations, 
    use_highly_variable=True,
    cell_type_key="cytokine_coarse", 
    use_weights=True,
    lam=0.1, # varies depending on data and gene sets, try between 0.5 and 0.001
    delta=0.001, 
    kappa=None,
    rho=0.001, 
    use_cell_types=True,
    n_top_vals=50,
    label_factors=True, 
    overlap_threshold=0.2,
    clean_gs = True, 
    min_gs_num = 3,
    num_epochs=num_epochs #here running only 2 epochs for time reasons, we recommend 10,000 epochs for most datasets
)

# adata.uns['SPECTRA_factors'].shape

#so you can construct a dataframe for the factor gene weights

#include cell type specificity as a prefix into the index
# index_labels = adata.uns['SPECTRA_overlap'].index
# gene_weights = pd.DataFrame(adata.uns['SPECTRA_factors'], 
#                             index= index_labels,
#                             columns=adata.var[adata.var['spectra_vocab']].index)
# gene_weights

#add cell scores to obs
# cell_scores = adata.obsm['SPECTRA_cell_scores'][:,0].astype(float)
# adata.obs[factor_of_interest] = cell_scores
# sc.pl.umap(adata,color=factor_of_interest,s=30,vmax=np.quantile(cell_scores,0.98))
#save the results
from time import time 

cur_time = time()

adata.write_h5ad(f'/data/peer/rsaha/supervised-spectra/scripts/results/cui23_processed_spectrav2_example_20241216_spectra_epochs_{num_epochs}-time_{cur_time}.h5ad')

# save the model as pickle
import pickle

import os

output_dir = '/data/peer/rsaha/supervised-spectra/scripts/results'
# os.makedirs(output_dir, exist_ok=True)

with open(f'{output_dir}/cui23_processed_spectrav2_example_20241216_spectra_epochs_{num_epochs}-time_{cur_time}.pickle', 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)