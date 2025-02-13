import numpy as np
import torch
from collections import OrderedDict
from opt_einsum import contract
from tqdm import tqdm
from Spectra import Spectra_util
import pandas as pd

### Class for SPECTRA model
from Spectra.initialization import *
from Spectra.Spectra_core import *


class SPECTRA_Model_Base:
    def __init__(
        self,
        X,
        labels,
        L,
        vocab=None,
        gs_dict=None,
        use_weights=True,
        adj_matrix=None,
        weights=None,
        lam=0.01,
        delta=0.001,
        kappa=None,
        rho=0.001,
        use_cell_types=False,
    ):
        self.L = L
        self.lam = lam
        self.delta = delta
        self.kappa_val = kappa
        self.rho_val = rho
        self.use_cell_types = use_cell_types
        self.vocab = vocab

        # If gene set dictionary is provided, process it to obtain adj_matrix and weights.
        if gs_dict is not None:
            gene2id = dict((v, idx) for idx, v in enumerate(vocab))
            adj_matrix, weights = Spectra_util.process_gene_sets(
                gs_dict=gs_dict,
                gene2id=gene2id,
                use_cell_types=use_cell_types,
                weighted=use_weights,
            )

        # Instantiate the internal SPECTRA model.
        self.internal_model = self.__get_spectra(
            X=X,
            labels=labels,
            adj_matrix=adj_matrix,
            L=L,
            weights=weights,
            lam=lam,
            delta=delta,
            kappa=kappa,
            rho=rho,
            use_cell_types=use_cell_types,
        )

        # Placeholders for parameters to be stored after training.
        self.cell_scores = None
        self.factors = None
        self.B_diag = None
        self.eta_matrices = None
        self.gene_scalings = None
        self.rho = None
        self.kappa = None

    def train(self, 
              X, 
              labels=None, 
              lr_schedule=[1.0, 0.5, 0.1, 0.01, 0.001, 0.0001], 
              num_epochs=10000, 
              verbose=False):
        """
        Train the internal SPECTRA model.
        
        Implements an adaptive learning rate schedule; if the loss does not decrease over epochs,
        the learning rate is updated accordingly.
        
        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Expression data for training.
        labels : np.ndarray, optional
            Cell type labels (required if use_cell_types is True).
        lr_schedule : list of float, optional
            A list of learning rate values to cycle through.
        num_epochs : int, optional
            Maximum number of training epochs.
        verbose : bool, optional
            If True, prints messages when the learning rate is updated.
        """
        opt = torch.optim.Adam(self.internal_model.parameters(), lr=lr_schedule[0])
        counter = 0
        last = float("inf")
        for i in tqdm(range(num_epochs)):
            opt.zero_grad()
            loss = self._get_loss(X, labels)
            loss.backward()
            opt.step()
            if loss.item() >= last:
                counter += 1
                if int(counter / 3) >= len(lr_schedule):
                    break
                if counter % 3 == 0:
                    opt = torch.optim.Adam(self.internal_model.parameters(), lr=lr_schedule[int(counter / 3)])
                    if verbose:
                        print("UPDATING LR TO " + str(lr_schedule[int(counter / 3)]))
            last = loss.item()
        self._store_parameters(labels)

    def save(self, fp):
        torch.save(self.internal_model.state_dict(), fp)

    def load(self, fp, labels=None):
        self.internal_model.load_state_dict(torch.load(fp))
        self._store_parameters(labels)

    def __get_spectra(self,
        X,
        labels,
        adj_matrix,
        L,
        weights,
        lam,
        delta,
        kappa,
        rho,
        use_cell_types):
        if use_cell_types:
            return SpectraCellType(X, labels, adj_matrix, L, weights, lam, delta, kappa, rho)
        else:
            return SpectraNoCellType(X, adj_matrix, L, weights, lam, delta, kappa, rho)

    def return_eta_diag(self):
        return self.B_diag

    def return_cell_scores(self):
        return self.cell_scores

    def return_factors(self):
        return self.factors

    def return_eta(self):
        return self.eta_matrices

    def return_rho(self):
        return self.rho

    def return_kappa(self):
        return self.kappa

    def return_gene_scalings(self):
        return self.gene_scalings

    # Abstract methods to be implemented by subclasses.
    def _get_loss(self, X, labels):
        raise NotImplementedError("Subclasses must implement _get_loss.")

    def _store_parameters(self, labels):
        raise NotImplementedError("Subclasses must implement _store_parameters.")

    def initialize(self, annotations, word2id, W, init_scores=None, val=25):
        raise NotImplementedError("Subclasses must implement initialize.")

    def return_graph(self, ct="global"):
        raise NotImplementedError("Subclasses must implement return_graph.")

    def matching(self, markers, gene_names_dict, threshold=0.4):
        raise NotImplementedError("Subclasses must implement matching.")


class SPECTRA_Model_CellType(SPECTRA_Model_Base):
    def _get_loss(self, X, labels):
        # For cell type mode, the loss uses labels.
        assert labels is not None and len(labels) == X.shape[0]
        return self.internal_model.loss(X, labels)

    def _store_parameters(self, labels):
        """
        Extract and store derived parameters from the internal model for cell type-specific mode.
        
        After training, this method computes and stores the following:
        - Cell scores (loading matrix)
        - Factor matrices (theta)
        - Interaction matrices (eta) and their diagonals
        - Gene scaling values
        - Estimated kappa and rho values
        
        Parameters
        ----------
        labels : np.ndarray
            Cell type labels used to aggregate and compute factor and cell score matrices.
        """
        model = self.internal_model
        # Compute combined cell scores and factor matrices.
        k = sum(list(model.L.values()))
        out = np.zeros((model.n, k))
        global_idx = model.L["global"]

        tot = global_idx
        f = ["global"] * model.L["global"]
        for cell_type in model.cell_types:
            alpha = torch.exp(model.alpha[cell_type]).detach().numpy()
            out[labels == cell_type, :global_idx] = alpha[:, :global_idx]
            out[labels == cell_type, tot:tot + model.L[cell_type]] = alpha[:, global_idx:]
            tot += model.L[cell_type]
            f += [cell_type] * model.L[cell_type]

        out2 = np.zeros((k, model.p))
        theta_global = torch.softmax(model.theta["global"], dim=1)
        theta = theta_global.detach().numpy().T
        tot = theta.shape[0]
        out2[0:theta.shape[0], :] = theta
        for cell_type in model.cell_types:
            theta_ct = torch.softmax(model.theta[cell_type], dim=1)
            theta = theta_ct.detach().numpy().T
            out2[tot:tot + theta.shape[0], :] = theta
            tot += theta.shape[0]
        factors = out2

        lst = []
        for i in range(len(f)):
            ct = f[i]
            scaled = factors[i, :] * (torch.sigmoid(model.gene_scaling[ct]).detach() + model.delta).numpy()
            lst.append(scaled)
        scaled = np.array(lst)
        new_factors = scaled / (scaled.sum(axis=0, keepdims=True) + 1.0)
        cell_scores = out * scaled.mean(axis=1).reshape(1, -1)

        self.cell_scores = cell_scores
        self.factors = new_factors
        self.B_diag = self.__B_diag()
        self.eta_matrices = self.__eta_matrices()
        self.gene_scalings = {ct: torch.sigmoid(model.gene_scaling[ct]).detach().numpy() for ct in model.gene_scaling.keys()}
        self.rho = {ct: torch.sigmoid(model.rho[ct]).detach().numpy() for ct in model.rho.keys()}
        self.kappa = {ct: torch.sigmoid(model.kappa[ct]).detach().numpy() for ct in model.kappa.keys()}

    def __B_diag(self):
        model = self.internal_model
        k = sum(list(model.L.values()))
        out = np.zeros(k)
        Bg = torch.sigmoid(model.eta["global"])
        Bg = 0.5 * (Bg + Bg.T)
        B = torch.diag(Bg).detach().numpy()
        tot = B.shape[0]
        out[0:B.shape[0]] = B
        for cell_type in model.cell_types:
            Bg = torch.sigmoid(model.eta[cell_type])
            Bg = 0.5 * (Bg + Bg.T)
            B = torch.diag(Bg).detach().numpy()
            out[tot:tot + B.shape[0]] = B
            tot += B.shape[0]
        return out

    def __eta_matrices(self):
        model = self.internal_model
        eta = OrderedDict()
        Bg = torch.sigmoid(model.eta["global"])
        Bg = 0.5 * (Bg + Bg.T)
        eta["global"] = Bg.detach().numpy()
        for cell_type in model.cell_types:
            Bg = torch.sigmoid(model.eta[cell_type])
            Bg = 0.5 * (Bg + Bg.T)
            eta[cell_type] = Bg.detach().numpy()
        return eta

    def initialize(self, annotations, word2id, W, init_scores=None, val=25):
        if init_scores is None:
            init_scores = compute_init_scores(annotations, word2id, torch.Tensor(W))
        gs_dict = OrderedDict()
        for ct in annotations.keys():
            mval = max(self.L[ct] - 1, 0)
            sorted_init_scores = sorted(init_scores[ct].items(), key=lambda x: x[1])
            sorted_init_scores = sorted_init_scores[-mval:]
            names = set([k[0] for k in sorted_init_scores])
            lst_ct = []
            for key in annotations[ct].keys():
                if key in names:
                    words = annotations[ct][key]
                    idxs = [word2id[word] for word in words if word in word2id]
                    lst_ct.append(idxs)
            gs_dict[ct] = lst_ct
        self.internal_model.initialize(gene_sets=gs_dict, val=val)

    def return_graph(self, ct="global"):
        model = self.internal_model
        eta = torch.sigmoid(model.eta[ct])
        eta = 0.5 * (eta + eta.T)
        theta = torch.softmax(model.theta[ct], dim=1)
        mat = contract("il,lj,kj->ik", theta, eta, theta).detach().numpy()
        return mat

    def matching(self, markers, gene_names_dict, threshold=0.4):
        markers = pd.DataFrame(markers)
        matches = []
        jaccards = []
        for i in range(markers.shape[0]):
            max_jacc = 0.0
            best = ""
            for key in gene_names_dict.keys():
                for gs in gene_names_dict[key].keys():
                    t = gene_names_dict[key][gs]
                    jacc = Spectra_util.overlap_coefficient(list(markers.iloc[i, :]), t)
                    if jacc > max_jacc:
                        max_jacc = jacc
                        best = gs
            matches.append(best)
            jaccards.append(max_jacc)
        output = [matches[j] if jaccards[j] > threshold else "0" for j in range(markers.shape[0])]
        return np.array(output)


class SPECTRA_Model_NoCellType(SPECTRA_Model_Base):
    def _get_loss(self, X, labels=None):
        return self.internal_model.loss(X)

    def _store_parameters(self, labels=None):
        model = self.internal_model
        theta_ct = torch.softmax(model.theta, dim=1)
        theta = theta_ct.detach().numpy().T
        alpha = torch.exp(model.alpha).detach().numpy()
        out = alpha
        factors = theta
        scaled = factors * (torch.sigmoid(model.gene_scaling.detach()) + model.delta).numpy().reshape(1, -1)
        new_factors = scaled / (scaled.sum(axis=0, keepdims=True) + 1.0)
        self.factors = new_factors
        self.cell_scores = out * scaled.mean(axis=1).reshape(1, -1)
        Bg = torch.sigmoid(model.eta)
        Bg = 0.5 * (Bg + Bg.T)
        self.B_diag = torch.diag(Bg).detach().numpy()
        self.eta_matrices = Bg.detach().numpy()
        self.gene_scalings = torch.sigmoid(model.gene_scaling.detach()).numpy()
        self.rho = torch.sigmoid(model.rho.detach()).numpy()
        self.kappa = torch.sigmoid(model.kappa.detach()).numpy()

    def initialize(self, annotations, word2id, W, init_scores=None, val=25):
        if init_scores is None:
            init_scores = compute_init_scores_noct(annotations, word2id, torch.Tensor(W))
        lst = []
        mval = max(self.L - 1, 0)
        sorted_init_scores = sorted(init_scores.items(), key=lambda x: x[1])
        sorted_init_scores = sorted_init_scores[-mval:]
        names = set([k[0] for k in sorted_init_scores])
        for key in annotations.keys():
            if key in names:
                words = annotations[key]
                idxs = [word2id[word] for word in words if word in word2id]
                lst.append(idxs)
        self.internal_model.initialize(gs_list=lst, val=val)

    def return_graph(self, ct="global"):
        model = self.internal_model
        eta = torch.sigmoid(model.eta)
        eta = 0.5 * (eta + eta.T)
        theta = torch.softmax(model.theta, dim=1)
        mat = contract("il,lj,kj->ik", theta, eta, theta).detach().numpy()
        return mat

    def matching(self, markers, gene_names_dict, threshold=0.4):
        markers = pd.DataFrame(markers)
        matches = []
        jaccards = []
        for i in range(markers.shape[0]):
            max_jacc = 0.0
            best = ""
            for key in gene_names_dict.keys():
                t = gene_names_dict[key]
                jacc = Spectra_util.overlap_coefficient(list(markers.iloc[i, :]), t)
                if jacc > max_jacc:
                    max_jacc = jacc
                    best = key
            matches.append(best)
            jaccards.append(max_jacc)
        output = [matches[j] if jaccards[j] > threshold else "0" for j in range(markers.shape[0])]
        return np.array(output)


class SPECTRA_Model:
    """
    Top-level wrapper that dispatches to the appropriate specialized model based on use_cell_types.
    The external API remains the same.
    """
    def __init__(
        self,
        use_cell_types=True,
        **kwargs
    ):
        if use_cell_types:
            self.model = SPECTRA_Model_CellType(
                use_cell_types=True,
                **kwargs
            )
        else:
            self.model = SPECTRA_Model_NoCellType(
                use_cell_types=False,
                **kwargs
            )
        self.use_cell_types = use_cell_types

    def __getattr__(self, name):
        model = object.__getattribute__(self, "model")
        return getattr(model, name)

