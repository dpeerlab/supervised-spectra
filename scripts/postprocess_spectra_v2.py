"""
File: postprocess_spectra.py
Author(s): Suhani Balachandran, Sam Rose
Date: June 2023
Purpose: script to automate post-processing and analyzing Spectra results
"""

# IMPORTS
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import sys
import pickle
import json
from opt_einsum import contract
from scipy import sparse
import torch
import Spectra as spc
from Spectra import Spectra_util as spc_tl

# ---------------------------------------------------------------------
# TASK 0: MATCH FORMATTING FROM SPECTRA RUN SCRIPT
def get_vocab(var_path, gene_key = 'ensembl_id'):
    # gene set dictionary capitalize, get vocab
    v = pd.read_csv(var_path)
    vocab = pd.DataFrame(list(v[list(v['spectra_vocab'])][gene_key]), columns=[gene_key])
    return vocab

def get_vocab_from_adata(adata, geneset_annotations):
    # add column to adata.var for vocab that checks if gene is highly variable or in one of the gene sets
    # get union of all genes in gene sets
    # if geneset annotations is a dict of dicts, get all genes in all gene sets

    geneset_genes = []
    
    for i,v in geneset_annotations.items():
        if type(v) == dict:
            for k in v.keys():
                geneset_genes += v[k]
        else:
            geneset_genes += v
    geneset_genes = list(np.unique(geneset_genes))

    adata.var['spectra_vocab'] = adata.var_names.isin(geneset_genes) | adata.var['highly_variable']
    
    # vocab will be pd series of gene names
    # same length as number of genes input to spectra and same order
    # the order is based on true false subset of adata.var_names with order preserved
    vocab = pd.Series(adata.var_names[adata.var['spectra_vocab']])
    
    return(vocab)

# ---------------------------------------------------------------------
# TASK 1: GET CELL TYPES FOR EACH FACTOR
# Return markers index for factors
def return_markers_idx(factor_matrix, n_top_vals):
    idx_matrix = np.argsort(factor_matrix,axis = 1)[:,::-1][:,:n_top_vals]
    return(idx_matrix)

# Find the names of the top n highest scoring genes for each factor
def return_markers(factor_matrix, vocab, n_top_vals = 50):
    # PARAMS:
    # factor_matrix: model.factors (model is from pickle.load of Spectra results)
    # vocab: pd df of csv file of Spectra vocab saved right after run
    # n_top_vals: value passed to the n_top_vals param when running Spectra
    # RETURNS: df of 50 most important genes for each factor (factor x genes)

    idx_matrix = return_markers_idx(factor_matrix, n_top_vals)

    df = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df.iloc[i,j] = vocab.iloc[idx_matrix[i,j]]
            
    return df.values

# Add gene_scalings, cell_scores, and SPECTRA_vocab from the model obj
# to adata obj in the necessary format to be used by spectra_util functions
def add_model_to_anndata(adata, model, vocab):
    # PARAMS:
    # adata: AnnData object
    # model: model object of Spectra results
    # vocab: pd df of csv file of Spectra vocab saved right after run
    ## this is a series of gene names the same length as the number of genes input to spectra

    adata.uns['SPECTRA_factors'] = model.factors
    adata.obsm['SPECTRA_cell_scores'] = model.cell_scores
    # add gene names to uns and boolean index to var
    adata.uns['SPECTRA_vocab'] = vocab
    adata.var['spectra_vocab'] = adata.var_names.isin(vocab)
    adata.uns['SPECTRA_markers'] = return_markers(adata.uns['SPECTRA_factors'], vocab, n_top_vals = 50)
    return adata

# Get cell type specific factors, cell scores, gene weights
def cell_types_for_factor(adata, obs_key='cell_type', cellscore_obsm_key='SPECTRA_cell_scores'):
    # PARAMS: adata: AnnData object
    # RETURNS: gene_weights: pd df

    # get the celltype for each factor
    celltypes = spc_tl.get_factor_celltypes(adata, obs_key=obs_key, cellscore_obsm_key=cellscore_obsm_key)
    # include cell type specificity as a prefix into the index
    index_labels = [celltypes[x]+'_'+str(x) for x in range(adata.uns['SPECTRA_factors'].shape[0])]
    gene_weights = pd.DataFrame(adata.uns['SPECTRA_factors'], index = index_labels ,
                            columns= adata.var_names[adata.var['spectra_vocab']])
    # get the cell scores as a df
    column_labels = [celltypes[x]+'_'+str(x) for x in range(adata.obsm[cellscore_obsm_key].shape[1])]
    cell_scores = pd.DataFrame(adata.obsm[cellscore_obsm_key], 
                                index= adata.obs_names, columns=column_labels)
    # append factor scores as columns to obs
    adata.obs = pd.concat([adata.obs, cell_scores], axis = 1)

    return (cell_scores, gene_weights, adata)

# ---------------------------------------------------------------------
# TASK 2: ASSIGN FACTORS TO GENE PROGRAMS
# Check for geneset that has highest overlap coefficient with each factor
def annotate_factors(adata, geneset_anno, factor_names):
    # PARAMS:
    # adata: AnnData object
    # geneset_anno: geneset dictionary
    # factor_names: cell-type labeled factor names, list
    # RETURNS: factor_geneset_anno: pd df of factor, best_match_geneset, oc

    factor_geneset_anno = pd.DataFrame({'best_match_geneset': [""] * len(factor_names),
                             'overlap_coef': [0]*len(factor_names)}, index=factor_names)
    # remove trailing _[0-9]+$ from factor names for celltype
    factor_geneset_anno['celltype'] = factor_geneset_anno.index.str.replace(r'_*\d+$', '', regex = True)
    
    # if geneset_anno is not a nested dict, make it one with the top level global
    # if (factor_geneset_anno['celltype'].unique() == 'global').all() and (len(factor_geneset_anno['celltype'].unique()) == 1):
    # geneset_anno = {'global': geneset_anno}
    # make dict of marker genes for each factor
    fac_gene_dict = {}
    for i, fac in enumerate(factor_names):
        fac_gene_dict[fac] = [str(x) for x in adata.uns['SPECTRA_markers'][i,:]]
    # annotate factors by celltype
    for celltype in factor_geneset_anno['celltype'].unique():
    # filter the original df to work per celltype
        df = factor_geneset_anno[factor_geneset_anno['celltype'] == celltype]
        # for each factor in that celltype
        for fac in df.index:
            fac_genes = fac_gene_dict[fac]
            # compare to all reference genesets for that celltype
            for g in geneset_anno[celltype].keys():
                oc = spc_tl.overlap_coefficient(fac_genes, geneset_anno[celltype][g])
                if oc > factor_geneset_anno.loc[fac, 'overlap_coef']:
                    factor_geneset_anno.loc[fac, 'best_match_geneset'] = g
                    factor_geneset_anno.loc[fac, 'overlap_coef'] = oc
    # mark unannotated factors
    factor_geneset_anno['best_match_geneset'] = factor_geneset_anno['best_match_geneset'].replace('','unannotated')
    return factor_geneset_anno

# ---------------------------------------------------------------------
# TASK 3: CALC RECONSTRUCTION ERROR (holdout loss) & COHERENCE PER CELL TYPE
# Calculate coherence
def mimno_coherence_single(w1,w2,W):
    dw1 = W[:, w1] > 0
    dw2 = W[:, w2] > 0

    dw1w2 = (dw1 & dw2).float().sum() 
    dw1 = dw1.float().sum() 
    dw2 = dw2.float().sum() 
    if ((dw1 == 0)|(dw2 == 0)): return(-.1*np.inf)
    return ((dw1w2 + 1)/(dw2)).log()
def mimno_coherence_2011(words, W, vocab): 
    zero_vals = []
    
    score = 0
    V = len(words)
    for i in range(1, V):
        for j in range(i): 
            coh = mimno_coherence_single(words[i], words[j], W)
            if (coh == -np.inf):
                for x in (i,j):
                    if (sum(W[:,words[x]]) == 0): zero_vals.append(vocab[words[x]])
            else: score+=coh
    denom = V*(V-1)/2

    return(score/denom)

# Calculate the holdout loss per cell type
## TODO: add only_global option
# - test this
def holdout_loss(model, cell_type, celltype_labels, global_vals):
    # PARAMS:
    # model: model obj
    # cell_type: 1 indiv unique cell type label
    # celltype_labels: list of all labels, in order
    # global_vals: (theta_global, gene_scaling_global)
    # only_global: bool, whether only global factors are used
    # RETURNS: array of losses

    if type(model.theta) == torch.nn.modules.container.ParameterDict:
        gene_scaling_ct = model.gene_scaling[cell_type].exp()/(1.0 + model.gene_scaling[cell_type].exp())
        theta_global = contract('jk,j->jk',global_vals[0], global_vals[1] + model.delta)
        theta_ct = contract('jk,j->jk', torch.softmax(model.theta[cell_type], dim = 1), gene_scaling_ct + model.delta)
        theta = torch.cat((theta_global, theta_ct),1)
        alpha = torch.exp(model.alpha[cell_type])
        
        recon = contract('ik,jk->ij', alpha, theta) 
        X_c = model.X[celltype_labels == cell_type]
        tot_loss = model.lam * (-1.0*(torch.xlogy(X_c,recon) - recon).sum())
        
        lst = []
        loss_cf = 0.0
        for j in range(theta.shape[1]):
            mask = torch.ones(theta.shape[1])
            mask[j] = 0
            mask = mask.reshape(1,-1)
            recon = contract('ik,jk->ij', alpha, theta*mask) 
            loss_cf = model.lam * (-1.0*(torch.xlogy(X_c,recon) - recon).sum())
            lst.append(((loss_cf - tot_loss)/tot_loss).detach().numpy().item())
    else:
        
        theta = contract('jk,j->jk',global_vals[0], global_vals[1] + model.delta)
        alpha = torch.exp(model.alpha)
        
        recon = contract('ik,jk->ij', alpha, theta) 
        X_c = model.X
        tot_loss = model.lam * (-1.0*(torch.xlogy(X_c,recon) - recon).sum())
        
        lst = []
        loss_cf = 0.0
        for j in range(theta.shape[1]):
            mask = torch.ones(theta.shape[1])
            mask[j] = 0
            mask = mask.reshape(1,-1)
            recon = contract('ik,jk->ij', alpha, theta*mask) 
            loss_cf = model.lam * (-1.0*(torch.xlogy(X_c,recon) - recon).sum())
            lst.append(((loss_cf - tot_loss)/tot_loss).detach().numpy().item())
        
    return(np.array(lst))

# Get information and holdout scores for all cell types
def get_all_scores(model, idx_matrix, celltype_labels, adata, factor_names):
    lst = []
    for cell_type in np.unique(celltype_labels):
        lst.append([mimno_coherence_2011(list(idx_matrix[j,:]), model.X[celltype_labels == cell_type], list(adata.uns['SPECTRA_vocab'])) for j in range(idx_matrix.shape[0])])
    return(lst)

# Get information and holdout scores for all cell types
def get_all_scores_holdout(model, celltype_labels):
    
    # set condition whether only global factors are used
    if type(model.theta) == torch.nn.modules.container.ParameterDict:
        theta_global = torch.softmax(model.theta["global"], dim = 1)
        gene_scaling_global = model.gene_scaling["global"].exp()/(1.0 + model.gene_scaling["global"].exp())
        global_vals = (theta_global, gene_scaling_global)

        lst = [holdout_loss(model, cell_type, celltype_labels, global_vals) for cell_type in np.unique(celltype_labels)]
    else:
        theta_global = torch.softmax(model.theta, dim = 1)
        gene_scaling_global = model.gene_scaling.exp()/(1.0 + model.gene_scaling.exp())
        global_vals = (theta_global, gene_scaling_global)
        lst = [holdout_loss(model, cell_type, celltype_labels, global_vals) for cell_type in np.unique(celltype_labels)]
    return(lst)

# ---------------------------------------------------------------------
# TASK 4: OUTPUT
# To csv
def combine_info(adata, model, gene_weights, factor_geneset_anno, cell_type_key = 'spectra_key'):
    # PARAMS:
    # adta: AnnData objection
    # gene_weights: result of cell_types_for_factor()
    # factor_geneset_anno: result of annotate_factors()
    # model: model obj of Spectra results
    # RETURNS: df of all information

    #model.internal_model.X = torch.tensor(adata[:,list(adata.uns['SPECTRA_vocab'].iloc[:,0])].X)
    model.internal_model.X = torch.tensor(adata[:,list(adata.uns['SPECTRA_vocab'])].X)
    info_list = get_all_scores(model.internal_model, \
            return_markers_idx(adata.uns['SPECTRA_factors'], n_top_vals = 30),\
            adata.obs[cell_type_key].values, adata, list(gene_weights.index))
    holdout_list = get_all_scores_holdout(model.internal_model, adata.obs[cell_type_key].values)
    cell_types = np.unique(adata.obs[cell_type_key].values)

    lst = []
    replace_pattern = r'_*\d+$'
    for i in range(len(cell_types)):
        # assume factor names are cell_type + '_' + factor number
        factor_cell_type = pd.Series(list(gene_weights.index.values)).str.replace(replace_pattern, '', regex = True)
        #factor_idx = pd.Series(gene_weights.index).str.contains(cell_types[i]) | pd.Series(gene_weights.index).str.contains('global')
        factor_idx = (factor_cell_type == cell_types[i]) | (factor_cell_type == 'global')
        factor_names = list(gene_weights.index[factor_idx])
        eta_vals = list(model.B_diag[factor_idx])
        geneset_match = factor_geneset_anno['best_match_geneset'].values[factor_idx]
        overlap_coef = factor_geneset_anno['overlap_coef'].values[factor_idx]
        info_scores = [np.exp(score).numpy() for score in pd.Series(info_list[i])[factor_idx]]
        for j in range(len(holdout_list[i])):
            #print(i, j, len(cell_types), len(factor_names), len(holdout_list[i]), len(info_scores), len(eta_vals), len(geneset_match), len(overlap_coef), flush=True)
            lst.append([cell_types[i], factor_names[j], holdout_list[i][j], info_scores[j], eta_vals[j], geneset_match[j], overlap_coef[j]])
    return(pd.DataFrame(lst, columns = ['cell_type', 'factor', 'importance', 'information', 'eta', 'geneset_match', 'overlap_coef']))

# To plots
def plot_importance_information(markers, cell_type, run_name, importance_thresh = 0.01, information_thresh = 0.1, eta_thresh = 0.01):
    # PARAMS:
    # markers: pd df, Spectra markers from combine_info()
    # celltype: str, one cell type label
    # run_name: str, part of path to save figures
    
    # Set the figure size
    plt.figure(figsize=(12, 8))

    # filter markers for only the cell type of interest
    markers = markers[markers['cell_type'] == cell_type]
    # Create a scatter plot of all the points
    plt.scatter(markers["information"], markers["importance"], c=markers["eta"], cmap="coolwarm", edgecolors="black")

    # Add a colorbar legend
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("eta")

    # Set the colors for the scatter plot based on the "eta_high" column
    plt.set_cmap("coolwarm")
    cmap = plt.get_cmap()
    norm = plt.Normalize(vmin=0, vmax=1)
    colors = cmap(norm(markers["eta"]))

    # Create a scatter plot of all the points with colored edges
    plt.scatter(markers["information"], markers["importance"], c=colors, s=50, edgecolors="black")

    # Add labels and title
    plt.xlabel("Information Score")
    plt.ylabel("Importance Score")
    plt.title("Scatter Plot of " + cell_type + " information score against importance score")

    # Label the points based on the "best_match" column if they pass the thresholds
    texts = []
    for i in range(len(markers)):
        if markers["information"].iloc[i] > information_thresh and markers["importance"].iloc[i] > importance_thresh and markers["eta"].iloc[i] > eta_thresh:
            texts.append(plt.text(markers["information"].iloc[i], markers["importance"].iloc[i], markers["geneset_match"].iloc[i]))

    plt.savefig("{}{}.png".format(run_name, cell_type))

# ---------------------------------------------------------------------
def main():
    # TASK 0: READ IN / LOAD RELEVANT FILES
    # note that this is written for pre-processed adata and geneset dict
    adata = sc.read_h5ad(sys.argv[1])

    # make sure to remove genes if that was done
    if sys.argv[7] != "None":
        adata = adata[:, ~adata.var_names.isin(pd.read_csv(sys.argv[7], header = None)[0].tolist())]
    
    # make sure to keep only cells within the fold if thats necessary
    if sys.argv[8] != 'None':
        with open(sys.argv[8], 'rb') as infile:
            fold_dict = json.load(infile)
        adata = adata[fold_dict[sys.argv[9]]].copy()
    
    # load model and gene set annotations
    with open(sys.argv[2], 'rb') as f:
        model = pickle.load(f)
    with open(sys.argv[4]) as f:
        geneset_annotations = json.load(f)

    run_name = sys.argv[2].split('/')[-1][:-7]
    results_dir = sys.argv[5]
    obs_key_use = sys.argv[6]
    
    # make sure annotations is filtered the same way it would be in run
    
    # geneset_annotations_f = spc_tl.check_gene_set_dictionary(adata, geneset_annotations['global'], obs_key = obs_key_use, use_cell_types = False)
    geneset_annotations_f = spc_tl.check_gene_set_dictionary(adata, geneset_annotations, obs_key = 'cytokine_coarse', global_key = 'global')
    
    
    # get vocab if not available
    vocab = get_vocab_from_adata(adata, geneset_annotations_f)

    # filter anndata for only genes in vocab to preserve memory
    adata = adata[:,list(vocab)].copy()

    # if .X is  sparse matrix, make it dense
    if sparse.issparse(adata.X):
        adata.X = adata.X.todense()
    

    
    # TASK 1: GET CELL TYPES FOR EACH FACTOR
    cell_scores, gene_weights, adata = cell_types_for_factor(add_model_to_anndata(adata, model, vocab), obs_key = obs_key_use)
    cell_scores.to_csv(results_dir + run_name + "cellscores.csv")
    gene_weights.to_csv(results_dir + run_name + "geneweights.csv")

    # TASK 2: ASSIGN FACTORS TO GENE PROGRAMS
    factor_geneset_anno = annotate_factors(adata, geneset_annotations_f, list(gene_weights.index))

    # TASKS 3/4: CALCULATIONS AND OUTPUT
    markers = combine_info(adata, model, gene_weights, factor_geneset_anno, cell_type_key = obs_key_use)
    markers.to_csv(results_dir + run_name + "markers.csv")
    #for c in np.unique(markers['cell_type']):
    #    plot_importance_information(markers, c, results_dir + run_name)
    
    #adata.write_csvs(results_dir)

# ---------------------------------------------------------------------
if (sys.argv[1] == "-h" or sys.argv[1] == "--help"):
    print("The following arguments are required for postprocess_spectra.py:\
          \n\t1. path to preprocessed adata file\
          \n\t2. path to model results .pickle file\
          \n\t3. path to model results var.csv file \
          \n\t4. path to properly formatted, preprocessed gene set annotations dictionary .json\
          \n\t5. path to results directory\
          \n\t6. obs key for cell type labels (default: spectra_key)\
          \n\t7. path to list of genes to remove (default: None)\
          \n\t8. path to fold dictionary (default: None)\
          \n\t9. fold number to subset to (default: None)\
          \n\nExample usage: (assuming it is being called from within the /data/peer/suhani/glasner dir)\
          \npython postprocess_spectra.py /data/adata.h5ad /spectra/model/model.pickle /spectra/model/var.csv /data/peer/sam/ref_data/ref_genomes/Homo_sapiens/gene_sets/cytosig/spectra_dicts/formatted_geneset.json /postprocess/ cell_type\
          \n\nNote that this is optimized for the LCA, including the use of 1 pre-processed, complete immune cell geneset dictionary and the cell type adata.obs field being named 'spectra_key'.\
          \nCheck that these facts are satisfied by your dataset before using.")
else: main()