import numpy as np
import torch
from opt_einsum import contract
from Spectra import Spectra_util
import torch.nn as nn

from torch.distributions.normal import Normal

### Class for SPECTRA model
from Spectra.initialization import *

class SpectraNoCellType(nn.Module):
    def __init__(self, X, adj_matrix, L, weights=None, lam=0.01, delta=0.001, kappa=None, rho=0.001):
        """
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Expression data with shape (n, p)
        adj_matrix : np.ndarray
            (p, p)-shaped adjacency matrix
        L : int
            Number of factors (global only)
        weights : np.ndarray or None
            (p, p)-shaped weight matrix; if None then no weights are used
        lam : float
            Lambda parameter (expression loss multiplier)
        delta : float
            Delta parameter (gene scaling lower bound)
        kappa : float or None
            Background edge rate; if None then kappa is estimated
        rho : float or None
            Background non-edge rate; if None then rho is estimated
        """
        super(SpectraNoCellType, self).__init__()
        self.delta = delta
        self.lam = lam
        self.L = L  # must be an int
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.__initialize_parameters(adj_matrix, weights, kappa, rho)

    def __initialize_parameters(self, adj_matrix, weights, kappa, rho):
        # Prepare matrices (the utility zeroes out the diagonal)
        self.adj_matrix = self.__prepare_adj_matrix(adj_matrix)
        self.adj_matrix_1m = self.__prepare_adj_matrix(1 - adj_matrix)
        self.weights = self.__prepare_weight_matrix(weights, adj_matrix)
        
        # Initialize global parameters as nn.Parameters
        self.theta = nn.Parameter(Normal(0.0, 1.0).sample([self.p, self.L]))
        self.alpha = nn.Parameter(Normal(0.0, 1.0).sample([self.n, self.L]))
        self.eta = nn.Parameter(Normal(0.0, 1.0).sample([self.L, self.L]))
        self.gene_scaling = nn.Parameter(Normal(0.0, 1.0).sample([self.p]))
        
        if kappa is None:
            self.kappa = nn.Parameter(Normal(0.0, 1.0).sample())
        else:
            self.kappa = torch.tensor(np.log(kappa / (1 - kappa)))
        
        if rho is None:
            self.rho = nn.Parameter(Normal(0.0, 1.0).sample())
        else:
            self.rho = torch.tensor(np.log(rho / (1 - rho)))

    def __prepare_adj_matrix(self, arr):
        # Convert the input array into a tensor with a zeroed diagonal.
        return torch.Tensor(Spectra_util.zero_out_diagonal(arr))

    def __prepare_weight_matrix(self, weights, adj_matrix):
        if weights is not None:
            return torch.Tensor(weights) - torch.Tensor(np.diag(np.diag(adj_matrix)))
        else:
            return adj_matrix

    def loss(self, X):
        """
        Compute the loss for the non-cell type (global) model.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Expression data.
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        X = torch.Tensor(X)
        theta = torch.softmax(self.theta, dim=1)
        eta = torch.sigmoid(self.eta)
        eta = 0.5 * (eta + eta.T)
        gene_scaling = torch.sigmoid(self.gene_scaling)
        kappa = torch.sigmoid(self.kappa)
        rho = torch.sigmoid(self.rho)
        alpha = torch.exp(self.alpha)

        # Adjust theta with gene scaling
        theta_ = contract("jk,j->jk", theta, gene_scaling + self.delta)
        recon = contract("ik,jk->ij", alpha, theta_)
        term1 = -1.0 * (torch.xlogy(X, recon) - recon).sum()

        if len(self.adj_matrix) > 0:
            mat = contract("il,lj,kj->ik", theta, eta, theta)
            term2 = -1.0 * torch.xlogy(
                self.adj_matrix * self.weights,
                (1.0 - rho) * (1.0 - kappa) * mat + (1.0 - rho) * kappa,
            ).sum()
            term3 = -1.0 * torch.xlogy(
                self.adj_matrix_1m,
                (1.0 - kappa) * (1.0 - rho) * (1.0 - mat) + rho,
            ).sum()
        else:
            term2 = 0.0
            term3 = 0.0

        return self.lam * term1 + term2 + term3

    def initialize(self, gs_list, val):
        """
        Initialize parameters using a list of gene sets.
        
        Parameters
        ----------
        gs_list : list
            A list of gene sets (each gene set is a list of gene indices).
        val : float
            Initialization value to be assigned.
        """
        assert self.L >= len(gs_list)
        count = 0
        for gene_set in gs_list:
            self.theta.data[:, count][gene_set] = val
            count += 1
        for i in range(self.L):
            self.eta.data[i, -1] = -val
            self.eta.data[-1, i] = -val
        self.theta.data[:, -1][self.adj_matrix.sum(axis=1) == 0] = val
        self.theta.data[:, -1][self.adj_matrix.sum(axis=1) != 0] = -val


class SpectraCellType(nn.Module):
    def __init__(self, X, labels, adj_matrix, L, weights=None, lam=0.01, delta=0.001, kappa=None, rho=0.001):
        """
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Expression data with shape (n, p)
        labels : np.ndarray
            (n, ) array of cell type labels.
        adj_matrix : dict
            Dictionary mapping cell types (and "global") to (p, p)-shaped adjacency matrices.
        L : dict
            Dictionary with keys "global" and cell types; values are number of factors.
        weights : dict or None
            Dictionary mapping cell types (and "global") to weight matrices; if None, weights are not used.
        lam : float
            Lambda parameter (expression loss multiplier)
        delta : float
            Delta parameter (gene scaling lower bound)
        kappa : float or None
            Background edge rate; if None then kappa is estimated.
        rho : float or None
            Background non-edge rate; if None then rho is estimated.
        """
        super(SpectraCellType, self).__init__()
        self.delta = delta
        self.lam = lam
        self.L = L  # expected to be a dict with key "global" plus one key per cell type
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.__initialize_parameters(labels, adj_matrix, weights, kappa, rho)

    def __initialize_parameters(self, labels, adj_matrix, weights, kappa, rho):
        # Prepare each cell type's adjacency matrices
        self.adj_matrix = {
            cell_type: self.__prepare_adj_matrix(mat) if len(mat) > 0 else [] 
            for cell_type, mat in adj_matrix.items()
        }
        self.adj_matrix_1m = {
            cell_type: self.__prepare_adj_matrix(1 - mat) if len(mat) > 0 else [] 
            for cell_type, mat in adj_matrix.items()
        }
        if weights is not None:
            self.weights = {
                cell_type: self.__prepare_weight_matrix(w, adj_matrix[cell_type]) if len(w) > 0 else [] 
                for cell_type, w in weights.items()
            }
        else:
            self.weights = self.adj_matrix

        # Identify unique cell types and count cells per type
        self.cell_types = np.unique(labels)
        self.cell_type_counts = {}
        for cell_type in self.cell_types:
            n_c = sum(labels == cell_type)
            self.cell_type_counts[cell_type] = n_c

        # Initialize parameters as ParameterDicts for global and cell-type-specific values
        self.theta = nn.ParameterDict()
        self.alpha = nn.ParameterDict()
        self.eta = nn.ParameterDict()
        self.gene_scaling = nn.ParameterDict()
        if kappa is None:
            self.kappa = nn.ParameterDict()
        if rho is None:
            self.rho = nn.ParameterDict()

        # Global parameters
        self.theta["global"] = nn.Parameter(Normal(0.0, 1.0).sample([self.p, self.L["global"]]))
        self.eta["global"] = nn.Parameter(Normal(0.0, 1.0).sample([self.L["global"], self.L["global"]]))
        self.gene_scaling["global"] = nn.Parameter(Normal(0.0, 1.0).sample([self.p]))
        if kappa is None:
            self.kappa["global"] = nn.Parameter(Normal(0.0, 1.0).sample())
        if rho is None:
            self.rho["global"] = nn.Parameter(Normal(0.0, 1.0).sample())

        # Cell type–specific parameters
        for cell_type in self.cell_types:
            self.theta[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample([self.p, self.L[cell_type]]))
            self.eta[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample([self.L[cell_type], self.L[cell_type]]))
            n_c = sum(labels == cell_type)
            # Note: for cell-type–specific alpha, the number of columns equals global factors + cell-type factors.
            self.alpha[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample([n_c, self.L["global"] + self.L[cell_type]]))
            self.gene_scaling[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample([self.p]))
            if kappa is None:
                self.kappa[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample())
            if rho is None:
                self.rho[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample())

        # If fixed kappa or rho are provided, store them as fixed tensors wrapped in ParameterDicts.
        if kappa is not None:
            kappa_dict = {}
            kappa_dict["global"] = torch.tensor(np.log(kappa / (1 - kappa)))
            for cell_type in self.cell_types:
                kappa_dict[cell_type] = torch.tensor(np.log(kappa / (1 - kappa)))
            self.kappa = nn.ParameterDict(kappa_dict)
        if rho is not None:
            rho_dict = {}
            rho_dict["global"] = torch.tensor(np.log(rho / (1 - rho)))
            for cell_type in self.cell_types:
                rho_dict[cell_type] = torch.tensor(np.log(rho / (1 - rho)))
            self.rho = nn.ParameterDict(rho_dict)

    def __prepare_adj_matrix(self, arr):
        return torch.Tensor(Spectra_util.zero_out_diagonal(arr))

    def __prepare_weight_matrix(self, weights, adj_matrix):
        if weights is not None:
            return torch.Tensor(weights) - torch.Tensor(np.diag(np.diag(adj_matrix)))
        else:
            return adj_matrix

    def loss(self, X, labels):
        """
        Compute the loss for the cell type–specific model.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Expression data.
        labels : np.ndarray
            Cell type labels for the cells.
        
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        X = torch.Tensor(X)
        loss = 0.0

        # Global parameters
        theta_global = torch.softmax(self.theta["global"], dim=1)
        eta_global = torch.sigmoid(self.eta["global"])
        eta_global = 0.5 * (eta_global + eta_global.T)
        gene_scaling_global = torch.sigmoid(self.gene_scaling["global"])
        kappa_global = torch.sigmoid(self.kappa["global"])
        rho_global = torch.sigmoid(self.rho["global"])

        # Loop over each cell type and compute its loss components.
        for cell_type in self.cell_types:
            kappa_ct = torch.sigmoid(self.kappa[cell_type])
            rho_ct = torch.sigmoid(self.rho[cell_type])
            gene_scaling_ct = torch.sigmoid(self.gene_scaling[cell_type])
            X_c = X[labels == cell_type]
            adj_matrix = self.adj_matrix[cell_type]
            weights = self.weights[cell_type]
            adj_matrix_1m = self.adj_matrix_1m[cell_type]
            theta_ct = torch.softmax(self.theta[cell_type], dim=1)
            eta_ct = torch.sigmoid(self.eta[cell_type])
            eta_ct = 0.5 * (eta_ct + eta_ct.T)
            
            theta_global_ = contract("jk,j->jk", theta_global, gene_scaling_global + self.delta)
            theta_ct_ = contract("jk,j->jk", theta_ct, gene_scaling_ct + self.delta)
            theta = torch.cat((theta_global_, theta_ct_), 1)
            alpha = torch.exp(self.alpha[cell_type])
            recon = contract("ik,jk->ij", alpha, theta)
            term1 = -1.0 * (torch.xlogy(X_c, recon) - recon).sum()
            
            if len(adj_matrix) > 0:
                mat = contract("il,lj,kj->ik", theta_ct, eta_ct, theta_ct)
                term2 = -1.0 * torch.xlogy(
                    adj_matrix * weights,
                    (1.0 - rho_ct) * (1.0 - kappa_ct) * mat + (1.0 - rho_ct) * kappa_ct,
                ).sum()
                term3 = -1.0 * torch.xlogy(
                    adj_matrix_1m,
                    (1.0 - kappa_ct) * (1.0 - rho_ct) * (1.0 - mat) + rho_ct,
                ).sum()
            else:
                term2 = 0.0
                term3 = 0.0
            
            loss += self.lam * term1 + (self.cell_type_counts[cell_type] / float(self.n)) * (term2 + term3)

        # Global graph loss
        global_adj = self.adj_matrix["global"]
        global_adj_1m = self.adj_matrix_1m["global"]
        global_weights = self.weights["global"]
        if len(global_adj) > 0:
            mat = contract("il,lj,kj->ik", theta_global, eta_global, theta_global)
            term2 = -1.0 * torch.xlogy(
                global_adj * global_weights,
                (1.0 - rho_global) * (1.0 - kappa_global) * mat + (1.0 - rho_global) * kappa_global,
            ).sum()
            term3 = -1.0 * torch.xlogy(
                global_adj_1m,
                (1.0 - kappa_global) * (1.0 - rho_global) * (1.0 - mat) + rho_global,
            ).sum()
            loss += term2 + term3

        return loss

    def initialize(self, gene_sets, val):
        """
        Initialize parameters using gene set annotations.
        
        Parameters
        ----------
        gene_sets : dict
            Dictionary mapping cell types (including "global") to collections of gene index sets.
        val : float
            Initialization value applied to the parameters.
        """
        for ct in self.cell_types:
            assert self.L[ct] >= len(gene_sets[ct])
            count = 0
            if self.L[ct] > 0:
                if len(self.adj_matrix[ct]) > 0:
                    for gene_set in gene_sets[ct]:
                        self.theta[ct].data[:, count][gene_set] = val
                        count += 1
                    for i in range(self.L[ct]):
                        self.eta[ct].data[i, -1] = -val
                        self.eta[ct].data[-1, i] = -val
                    self.theta[ct].data[:, -1][self.adj_matrix[ct].sum(axis=1) == 0] = val
                    self.theta[ct].data[:, -1][self.adj_matrix[ct].sum(axis=1) != 0] = -val

        assert self.L["global"] >= len(gene_sets["global"])
        count = 0
        for gene_set in gene_sets["global"]:
            self.theta["global"].data[:, count][gene_set] = val
            count += 1
        for i in range(self.L["global"]):
            self.eta["global"].data[i, -1] = -val
            self.eta["global"].data[-1, i] = -val
        self.theta["global"].data[:, -1][self.adj_matrix["global"].sum(axis=1) == 0] = val
        self.theta["global"].data[:, -1][self.adj_matrix["global"].sum(axis=1) != 0] = -val
