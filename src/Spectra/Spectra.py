import numpy as np
import torch
from Spectra import Spectra_util
import scipy
import pandas as pd
from pyvis.network import Network
import random

### Class for SPECTRA model
from Spectra.initialization import *
from Spectra.Spectra_core import (
    SpectraNoCellType,
    SpectraCellType,
)
from Spectra.Spectra_model import (
    SPECTRA_Model_Base,
    SPECTRA_Model_CellType,
    SPECTRA_Model_NoCellType,
    SPECTRA_Model
)


""" 
Public Functions 


    est_spectra():

    matching(): 

    graph_network():  

    markers


"""


def get_factor_celltypes(adata, obs_key, cellscore):
    """
    Assigns Spectra factors to cell types by analyzing the factor cell scores.
    Cell type specific factors will have zero cell scores except in their respective cell type

    adata: AnnData , object containing the Spectra output
    obs_key: str , column name in adata.obs containing the cell type annotations
    cellscore_obsm_key: str , key for adata.obsm containing the Spectra cell scores

    returns: dict , dictionary of {factor index : 'cell type'}
    """

    # get cellscores
    cell_scores_df = pd.DataFrame(cellscore)
    cell_scores_df["celltype"] = list(adata.obs[obs_key])

    # find global and cell type specific fators
    global_factors_series = (cell_scores_df.groupby("celltype").mean() != 0).all()
    global_factors = [
        factor
        for factor in global_factors_series.index
        if global_factors_series[factor]
    ]

    specific_factors = {}
    if len(global_factors) != (len(cell_scores_df.columns) - 1):
        specific_cell_scores = (
            (cell_scores_df.groupby("celltype").mean()).T[~global_factors_series].T
        )

        for i in set(cell_scores_df["celltype"]):
            specific_factors[i] = [
                factor
                for factor in specific_cell_scores.loc[i].index
                if specific_cell_scores.loc[i, factor]
            ]

        # inverse dict factor:celltype
        factors_inv = {}
        for i, v in specific_factors.items():
            for factor in v:
                factors_inv[factor] = i

        # add global

        for factor in global_factors:
            factors_inv[factor] = "global"

    else:
        factors_inv = {}
        for factor in global_factors:
            factors_inv[factor] = "global"

    return factors_inv


def est_spectra(
    adata,
    gene_set_dictionary,
    L=None,
    use_highly_variable=True,
    cell_type_key="cell_type_annotations",
    use_weights=True,
    lam=0.01,
    delta=0.001,
    kappa=None,
    rho=0.001,
    use_cell_types=True,
    n_top_vals=50,
    filter_sets=True,
    label_factors=True,
    clean_gs=True,
    min_gs_num=3,
    overlap_threshold=0.2,
    **kwargs,
):
    """

    Parameters
        ----------
        adata : AnnData object
            containing cell_type_key with log count data stored in .X
        gene_set_dictionary : dict or OrderedDict()
            maps cell types to gene set names to gene sets ; if use_cell_types == False then maps gene set names to gene sets ;
            must contain "global" key in addition to every unique cell type under .obs.<cell_type_key>
        L : dict, OrderedDict(), int , NoneType
            number of factors per cell type ; if use_cell_types == False then int. Else dictionary. If None then match factors
            to number of gene sets (recommended)
        use_highly_variable : bool
            if True, then uses highly_variable_genes
        cell_type_key : str
            cell type key, must be under adata.obs.<cell_type_key> . If use_cell_types == False, this is ignored
        use_weights : bool
            if True, edge weights are estimated based on graph structure and used throughout training
        lam : float
            lambda parameter of the model. weighs relative contribution of graph and expression loss functions
        delta : float
            delta parameter of the model. lower bounds possible gene scaling factors so that maximum ratio of gene scalings
            cannot be too large
        kappa : float or None
            if None, estimate background rate of 1s in the graph from data
        rho : float or None
            if None, estimate background rate of 0s in the graph from data
        use_cell_types : bool
            if True then cell type label is used to fit cell type specific factors. If false then cell types are ignored
        n_top_vals : int
            number of top markers to return in markers dataframe
        determinant_penalty : float
            determinant penalty of the attention mechanism. If set higher than 0 then sparse solutions of the attention weights
            and diverse attention weights are encouraged. However, tuning is crucial as setting too high reduces the selection
            accuracy because convergence to a hard selection occurs early during training [todo: annealing strategy]
        filter_sets : bool
            whether to filter the gene sets based on coherence
        label_factors : bool
            whether to label the factors by their cell type specificity and their overlap coefficient with the input marker genes
        clean_gs : bool
            if True cleans up the gene set dictionary to:
                1. checks that annotations dictionary cell type keys and adata cell types are identical.
                2. to contain only genes contained in the adata
                3. to contain only gene sets greater length min_gs_num
        min_gs_num : int
            only use if clean_gs True, minimum number of genes per gene set expressed in adata, other gene sets will be filtered out
        overlap_threshold: float
            minimum overlap coefficient to assign an input gene set label to a factor

        **kwargs : (num_epochs = 10000, lr_schedule = [...], verbose = False)
            arguments to .train(), maximum number of training epochs, learning rate schedule and whether to print changes in
            learning rate

     Returns: SPECTRA_Model object [after training]

     In place: adds 1. factors, 2. cell scores, 3. vocabulary, 4. markers, 5. overlap coefficient of markers vs input gene sets as attributes in .obsm, .var, .uns


    """

    # filter gene set dictionary
    if clean_gs:
        gene_set_dictionary = Spectra_util.check_gene_set_dictionary(
            adata,
            gene_set_dictionary,
            obs_key=cell_type_key,
            global_key="global",
            return_dict=True,
            min_len=min_gs_num,
            use_cell_types=use_cell_types,
        )
    if L is None:
        if use_cell_types:
            L = {}
            for key in gene_set_dictionary.keys():
                length = len(list(gene_set_dictionary[key].values()))
                L[key] = length + 1
        else:
            length = len(list(gene_set_dictionary.values()))
            L = length + 1
    # create vocab list from gene_set_dictionary
    lst = []
    if use_cell_types:
        for key in gene_set_dictionary:
            for key2 in gene_set_dictionary[key]:
                gene_list = gene_set_dictionary[key][key2]
                lst += gene_list
    else:
        for key in gene_set_dictionary:
            gene_list = gene_set_dictionary[key]
            lst += gene_list

    # lst contains all of the genes that are in the gene sets --> convert to boolean array
    bools = []
    for gene in adata.var_names:
        if gene in lst:
            bools.append(True)
        else:
            bools.append(False)
    bools = np.array(bools)

    if use_highly_variable:
        idx_to_use = (
            bools | adata.var.highly_variable.to_numpy()
        )  # take union of highly variable and gene set genes (todo: add option to change this at some point)
        X = adata.X[:, idx_to_use]
        vocab = adata.var_names[idx_to_use]
        adata.var["spectra_vocab"] = idx_to_use
    else:
        X = adata.X
        vocab = adata.var_names

    if use_cell_types:
        labels = adata.obs[cell_type_key].values
        for label in np.unique(labels):
            if label not in gene_set_dictionary:
                gene_set_dictionary[label] = {}
            if label not in L:
                L[label] = 1
    else:
        labels = None
    if type(X) == scipy.sparse.csr.csr_matrix:
        X = np.array(X.todense())
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    if filter_sets:
        if use_cell_types:
            new_gs_dict = {}
            init_scores = compute_init_scores(
                gene_set_dictionary, word2id, torch.Tensor(X)
            )
            for ct in gene_set_dictionary.keys():
                new_gs_dict[ct] = {}
                mval = max(L[ct] - 1, 0)
                sorted_init_scores = sorted(init_scores[ct].items(), key=lambda x: x[1])
                sorted_init_scores = sorted_init_scores[-1 * mval :]
                names = set([k[0] for k in sorted_init_scores])
                for key in gene_set_dictionary[ct].keys():
                    if key in names:
                        new_gs_dict[ct][key] = gene_set_dictionary[ct][key]
        else:
            init_scores = compute_init_scores_noct(
                gene_set_dictionary, word2id, torch.Tensor(X)
            )
            new_gs_dict = {}
            mval = max(L - 1, 0)
            sorted_init_scores = sorted(init_scores.items(), key=lambda x: x[1])
            sorted_init_scores = sorted_init_scores[-1 * mval :]
            names = set([k[0] for k in sorted_init_scores])
            for key in gene_set_dictionary.keys():
                if key in names:
                    new_gs_dict[key] = gene_set_dictionary[key]
        gene_set_dictionary = new_gs_dict
    else:
        init_scores = None

    spectra = SPECTRA_Model(
        use_cell_types=use_cell_types,
        X=X,
        labels=labels,
        L=L,
        vocab=vocab,
        gs_dict=gene_set_dictionary,
        use_weights=use_weights,
        lam=lam,
        delta=delta,
        kappa=kappa,
        rho=rho,
    )

    spectra.initialize(gene_set_dictionary, word2id, X, init_scores)

    spectra.train(X=X, labels=labels, **kwargs)

    adata.uns["SPECTRA_factors"] = spectra.factors
    adata.uns["SPECTRA_markers"] = return_markers(
        factor_matrix=spectra.factors, id2word=id2word, n_top_vals=n_top_vals
    )
    adata.uns["SPECTRA_L"] = L

    # label factors

    # transform input nested dictionary into a flat dictionary
    gene_set_dictionary_flat = {}

    for k, v in gene_set_dictionary.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                gene_set_dictionary_flat[k2] = v2
            is_global = False
        else:
            gene_set_dictionary_flat[k] = v
            is_global = True

    # labeling function
    if label_factors:
        # get cell type specificity of every factor
        if is_global == False:
            celltype_dict = get_factor_celltypes(
                adata, cell_type_key, cellscore=spectra.cell_scores
            )
            max_celltype = [
                celltype_dict[x] for x in range(spectra.cell_scores.shape[1])
            ]
        else:
            max_celltype = ["global"] * (spectra.cell_scores.shape[1])
        # get gene set with maximum overlap coefficient with top marker genes
        overlap_df = Spectra_util.label_marker_genes(
            adata.uns["SPECTRA_markers"],
            gene_set_dictionary_flat,
            threshold=overlap_threshold,
        )

        # create new column labels
        column_labels = []
        for i in range(len(spectra.cell_scores.T)):
            column_labels.append(
                str(i)
                + "-X-"
                + str(max_celltype[i])
                + "-X-"
                + str(list(overlap_df.index)[i])
            )

        overlap_df.index = column_labels
        adata.uns["SPECTRA_overlap"] = overlap_df

    adata.obsm["SPECTRA_cell_scores"] = spectra.cell_scores

    return spectra


def return_markers(factor_matrix, id2word, n_top_vals=100):
    idx_matrix = np.argsort(factor_matrix, axis=1)[:, ::-1][:, :n_top_vals]
    df = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df.iloc[i, j] = id2word[idx_matrix[i, j]]
    return df.values


def load_from_pickle(fp, adata, gs_dict, cell_type_key):
    """
    Load a pre-trained SPECTRA model from a pickle file.
    
    Uses the AnnData object to extract the required input data and annotations.
    
    Parameters
    ----------
    fp : str
        File path of the saved model state.
    adata : AnnData
        Annotated data object containing the spectra vocabulary and cell type annotations.
    gs_dict : dict
        Gene set dictionary.
    cell_type_key : str
        Key in adata.obs that contains the cell type annotations.
    
    Returns
    -------
    SPECTRA_Model
        The loaded SPECTRA model.
    """
    model = SPECTRA_Model(
        X=adata[:, adata.var["spectra_vocab"]].X,
        labels=np.array(adata.obs[cell_type_key]),
        L=adata.uns["SPECTRA_L"],
        vocab=adata.var_names[adata.var["spectra_vocab"]],
        gs_dict=gs_dict,
    )
    model.load(fp, labels=np.array(adata.obs[cell_type_key]))
    return model


def graph_network(adata, mat, gene_set, thres=0.20, N=50):
    """
    Create an interactive network visualization of the inferred graph using pyvis.
    
    Nodes represent genes; nodes corresponding to the input gene set are highlighted with a distinct color.
    Only the top N nodes (by summed association) are included, and edges are added if their weight exceeds thres.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object that contains the spectra vocabulary in adata.var.
    mat : np.ndarray
        Inferred graph matrix (e.g., factor interactions).
    gene_set : list
        List of gene names to highlight.
    thres : float, optional
        Threshold for edge inclusion.
    N : int, optional
        Maximum number of top nodes to include.
    
    Returns
    -------
    Network
        A pyvis Network object for interactive visualization.
    """
    vocab = adata.var_names[adata.var["spectra_vocab"]]
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=True,
    )
    net.barnes_hut()

    idxs = []
    for term in gene_set:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs, :].sum(axis=0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0
    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label=id2word[est], color="#00ff1e")
        else:
            net.add_node(count, label=id2word[est], color="#162347")
        count += 1

    inferred_mat = mat[ests, :][:, ests]
    for i in range(len(inferred_mat)):
        for j in range(i + 1, len(inferred_mat)):
            if inferred_mat[i, j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    return net


def graph_network_multiple(adata, mat, gene_sets, thres=0.20, N=50):
    """
    Create an interactive network visualization for multiple gene sets using pyvis.
    
    Nodes corresponding to different gene sets are visualized in different colors.
    The function aggregates all genes from the input gene sets and highlights them appropriately.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with spectra vocabulary.
    mat : np.ndarray
        Inferred graph matrix.
    gene_sets : list of lists
        List of gene sets (each gene set is a list of gene names).
    thres : float, optional
        Threshold for edge inclusion.
    N : int, optional
        Number of top nodes to include.
    
    Returns
    -------
    Network
        A pyvis Network object representing the graph.
    """
    gene_set = []
    for gs in gene_sets:
        gene_set += gs

    vocab = adata.var_names[adata.var["spectra_vocab"]]
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=True,
    )
    net.barnes_hut()
    idxs = []
    for term in gene_set:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs, :].sum(axis=0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0

    color_map = []
    for gene_set in gene_sets:
        random_color = [
            "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
        ]
        color_map.append(random_color[0])

    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label=id2word[est], color="#00ff1e")
        else:
            for i in range(len(gene_sets)):
                if id2word[est] in gene_sets[i]:
                    color = color_map[i]
                    break
            net.add_node(count, label=id2word[est], color=color)
        count += 1

    inferred_mat = mat[ests, :][:, ests]
    for i in range(len(inferred_mat)):
        for j in range(i + 1, len(inferred_mat)):
            if inferred_mat[i, j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    return net


def gene_set_graph(gene_sets):
    """
    Generate an interactive network graph for the provided gene sets.
    
    Each gene is represented as a node, and an edge is added between two genes if they belong to the same gene set.
    Each gene set is visualized with a distinct color.

    input
    [
    ["a","b", ... ],
    ["b", "d"],

    ...
    ]
    
    Parameters
    ----------
    gene_sets : list of lists
        Each inner list contains gene names belonging to a gene set.
    
    Returns
    -------
    Network
        A pyvis Network object representing the gene set graph.
    """

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=True,
    )
    net.barnes_hut()
    count = 0
    # create nodes
    genes = []
    for gene_set in gene_sets:
        genes += gene_set

    color_map = []
    for gene_set in gene_sets:
        random_color = [
            "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
        ]
        color_map.append(random_color[0])

    for gene in genes:
        for i in range(len(gene_sets)):
            if gene in gene_sets[i]:
                color = color_map[i]
                break
        net.add_node(gene, label=gene, color=color)

    for gene_set in gene_sets:
        for i in range(len(gene_set)):
            for j in range(i + 1, len(gene_set)):
                net.add_edge(gene_set[i], gene_set[j])

    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    return net
