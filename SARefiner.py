import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.models import LabelPropagation

from opengsl.data.dataset import Dataset
from opengsl.module.encoder import GNNEncoder_OpenGSL
from opengsl.utils import set_seed, accuracy
from opengsl.module.functional import normalize

class SimulatedAnnealingRefiner:
    def __init__(
        self,
        num_nodes,
        feats,
        l_train,
        l_val,
        t_max=200,
        alpha=50.0,
        beta=5.0,
        gamma=10.0,
        lambda_=0.1,
        delta=0.1,
        max_allowed_degree=20,
        add_fraction=0.05,
        device="cuda"
    ):
        self.num_nodes = int(num_nodes)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        feats = feats.astype(np.float64)
        self.feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)

        self.l_train = l_train.copy()
        self.l_val = l_val.copy()
        self.t_max = int(t_max)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.lambda_ = float(lambda_)
        self.delta = float(delta)
        self.max_allowed_degree = int(max_allowed_degree)
        self.device = device

        self.adj_sub = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)

        base = max(1, int(self.num_nodes * add_fraction))
        self.add_n_edges = base
        self.remove_n_edges = base

        self.time_step = 0
        self.prev_energy = 1e-12
        self.prev_acc = None
        self.earlystop_counter = 0

        sim = self.feats @ self.feats.T
        np.fill_diagonal(sim, -np.inf)
        self.sim = sim

    def _evaluate_energy(self, adj_sub, current_acc):
        adj_t = torch.tensor(adj_sub, dtype=torch.float32)
        edge_index = adj_t.nonzero(as_tuple=False).t().contiguous()

        y_tensor = torch.tensor(self.l_train, dtype=torch.float32).reshape(self.num_nodes, -1).to(self.device)
        if edge_index.numel() == 0:
            prop_labels = y_tensor
        else:
            model = LabelPropagation(num_layers=1, alpha=0.5)
            prop_labels = model(y=y_tensor, edge_index=edge_index.to(self.device))

        pred_y = prop_labels.squeeze().detach().cpu().numpy()
        ndcg_score = self._ndcg_at_k(pred_y, self.l_val, k=5)

        num_edges = int(adj_sub.sum())
        percentage_degree = num_edges / float(self.num_nodes * self.num_nodes)
        avg_degree = num_edges / float(self.num_nodes)

        out_degrees = adj_sub.sum(axis=1)
        degree_penalty = int(np.sum(np.maximum(0, out_degrees - self.max_allowed_degree)))

        energy = (self.alpha * float(current_acc)
                  + self.beta * float(np.mean(ndcg_score))
                  - self.gamma * float(percentage_degree)
                  - self.lambda_ * float(avg_degree)
                  - self.delta * float(degree_penalty))
        return energy, float(np.mean(ndcg_score)), percentage_degree, avg_degree, degree_penalty, out_degrees

    @staticmethod
    def _dcg_at_k(relevance, k):
        k = min(k, relevance.shape[1])
        relevance = np.asfarray(relevance)[:, :k]
        discounts = np.log2(np.arange(1, k + 1) + 1)
        return np.sum(relevance / discounts, axis=1)

    @staticmethod
    def _ndcg_at_k(predictions, ground_truth, k):
        actual_k_for_query = ground_truth.shape[1]
        k = min(k, actual_k_for_query)
        ranked_indices = np.argsort(predictions, axis=1)[:, ::-1]
        relevance = np.take_along_axis(ground_truth, ranked_indices, axis=1)
        dcg = SimulatedAnnealingRefiner._dcg_at_k(relevance, k)
        ideal_relevance = np.sort(ground_truth, axis=1)[:, ::-1]
        idcg = SimulatedAnnealingRefiner._dcg_at_k(ideal_relevance, k)
        return np.where(idcg > 0, dcg / idcg, 0.0)

    def _make_sparse_adj(self, adj_sub):
        edge_index = torch.tensor(adj_sub, dtype=torch.float32).nonzero(as_tuple=False).t().contiguous()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        adj_sparse = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1)),
            (self.num_nodes, self.num_nodes)
        ).to(self.device)
        return adj_sparse

    def step(self, current_acc):
        if self.time_step >= self.t_max:
            return self._make_sparse_adj(self.adj_sub)

        temperature = max(0.1, 1.0 - (self.time_step / float(self.t_max)))
        adj_copy = self.adj_sub.copy()

        improving = (self.prev_acc is None) or (current_acc > self.prev_acc + 1e-9)

        absent_mask = (self.adj_sub == 0).astype(float)
        np.fill_diagonal(absent_mask, 0.0)

        sim_scores = self.sim * absent_mask
        sim_flat = sim_scores.ravel()
        sim_flat = np.where(np.isfinite(sim_flat), sim_flat, -1e9)
        s = sim_flat - sim_flat.max()
        probs = np.exp(s / (temperature + 1e-9))
        probs[probs <= 0] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            avail_idx = np.where((absent_mask.ravel() > 0))[0]
            probs = np.zeros_like(sim_flat)
            if len(avail_idx) > 0:
                probs[avail_idx] = 1.0 / len(avail_idx)

        if improving:
            n_add = min(self.add_n_edges, int((absent_mask > 0).sum()))
            if n_add > 0:
                chosen = np.random.choice(len(probs), size=n_add, replace=False, p=probs)
                i_idx, j_idx = np.unravel_index(chosen, (self.num_nodes, self.num_nodes))
                self.adj_sub[i_idx, j_idx] = 1
        else:
            existing_edges = np.array(np.where(self.adj_sub > 0)).T
            if existing_edges.shape[0] > 0:
                n_remove = min(self.remove_n_edges, existing_edges.shape[0])
                remove_idx = np.random.choice(existing_edges.shape[0], size=n_remove, replace=False)
                to_remove = existing_edges[remove_idx]
                for (i, j) in to_remove:
                    self.adj_sub[int(i), int(j)] = 0

        energy, ndcg_mean, perc_deg, avg_deg, deg_pen, out_degrees = self._evaluate_energy(self.adj_sub, current_acc)
        max_deg = int(np.max(out_degrees)) if self.num_nodes > 0 else 0

        delta_e = energy - self.prev_energy
        if delta_e < 0 and (perc_deg < 0.1 or max_deg <= self.max_allowed_degree):
            p_accept = float(np.exp(delta_e / (temperature + 1e-9)))
            if np.random.rand() >= p_accept:
                self.adj_sub = adj_copy.copy()
        self.prev_energy = energy
        self.prev_acc = current_acc
        self.time_step += 1

        return self._make_sparse_adj(self.adj_sub)