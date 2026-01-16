import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.models import LabelPropagation

class PhysicsRefiner:
    """
    Refines a graph structure using a simulated annealing algorithm inspired by
    concepts from statistical mechanics.

    Key Physics Concepts Integrated:
    -  Ising Model for Homophily: An energy term rewards connections between nodes
        of the same class, analogous to spin alignment in ferromagnetic systems.
        The strength is controlled by the coupling constant `J`.
    -  Chemical Potential for Sparsity: A term `mu * num_edges` directly controls
        the number of edges (particles) in the system. A negative `mu` penalizes
        edges, promoting sparsity.
    -  Adaptive Cooling via Specific Heat: The cooling rate is adjusted based on
        the variance of the system's energy (analogous to specific heat). Cooling
        slows down during "phase transitions" (high variance), allowing for more
        careful exploration of critical states.
    """
    def __init__(
        self,
        num_nodes,
        feats,
        labels,
        l_train,
        l_val,
        t_max=200,
        alpha=50.0,
        beta=5.0,
        delta=0.1,
        J=10.0,
        mu=-15.0,
        isothermal_steps=20,
        base_adaptive_cooling_rate=0.97,
        max_allowed_degree=20,
        add_fraction=0.05,
        initial_temp=1.0,
        final_temp=0.01,
        lp_layers=1,
        lp_alpha=0.5,
        device="cuda"
    ):
        self.num_nodes = int(num_nodes)
        if isinstance(feats, torch.Tensor):
            feats = feats.detach().cpu().numpy()
        feats = feats.astype(np.float64)
        self.feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        self.l_train = l_train.copy()
        self.l_val = l_val.copy()
        self.device = device

        self.t_max = int(t_max)
        self.time_step = 0
        self.temperature = float(initial_temp)
        self.initial_temp = float(initial_temp)
        self.final_temp = float(final_temp)

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.max_allowed_degree = int(max_allowed_degree)
        self.prev_energy = -1e12
        self.prev_acc = None

        self.J = float(J)
        self.mu = float(mu)
        self.isothermal_steps = int(isothermal_steps)
        self.base_adaptive_cooling_rate = float(base_adaptive_cooling_rate)
        self.lp_layers = int(lp_layers)
        self.lp_alpha = int(lp_alpha)
        self.energy_history = []
        
        labels_i = labels.reshape(-1, 1)
        labels_j = labels.reshape(1, -1)
        self.label_agreement = (labels_i == labels_j).astype(np.float32) * 2 - 1
        np.fill_diagonal(self.label_agreement, 0)

        self.adj_sub = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)
        base = max(1, int(self.num_nodes * add_fraction))
        self.add_n_edges = base
        self.remove_n_edges = base
        sim = self.feats @ self.feats.T
        np.fill_diagonal(sim, -np.inf)
        self.sim = sim

    def _evaluate_energy(self, adj_sub, current_acc):
        """Calculates the total energy of a given graph configuration."""
        adj_t = torch.tensor(adj_sub, dtype=torch.float32)
        edge_index = adj_t.nonzero(as_tuple=False).t().contiguous()
        y_tensor = torch.tensor(self.l_train, dtype=torch.float32).reshape(self.num_nodes, -1).to(self.device)
        prop_labels = LabelPropagation(num_layers=self.lp_layers, alpha=self.lp_alpha)(y=y_tensor, edge_index=edge_index.to(self.device))
        pred_y = prop_labels.squeeze().detach().cpu().numpy()
        ndcg_score = self._ndcg_at_k(pred_y, self.l_val, k=2)

        out_degrees = adj_sub.sum(axis=1)
        degree_penalty = int(np.sum(np.maximum(0, out_degrees - self.max_allowed_degree)))
        
        homophily_energy = self.J * np.sum(adj_sub * self.label_agreement)
        
        num_edges = int(adj_sub.sum())
        sparsity_energy = self.mu * num_edges

        energy = (self.alpha * float(current_acc)
                  + self.beta * float(np.mean(ndcg_score))
                  - self.delta * float(degree_penalty)
                  + homophily_energy
                  + sparsity_energy)
                  
        return energy, float(np.mean(ndcg_score))

    def _propose_move(self, current_acc):
        """Proposes a new graph structure by adding or removing edges."""
        adj_proposal = self.adj_sub.copy()
        improving = (self.prev_acc is None) or (current_acc > self.prev_acc + 1e-9)

        if improving:
            absent_mask = (adj_proposal == 0).astype(np.float32)
            
            sim_scores = self.sim * absent_mask 
            
            np.fill_diagonal(sim_scores, -np.inf)

            sim_flat = sim_scores.ravel()
            
            if np.all(np.isneginf(sim_flat)):
                 valid_indices = np.where(absent_mask.ravel() > 0)[0]
                 probs = np.zeros_like(sim_flat)
                 if len(valid_indices) > 0:
                     probs[valid_indices] = 1.0 / len(valid_indices)
            else:
                 s = sim_flat - np.nanmax(sim_flat)
                 probs = np.exp(s / (self.temperature + 1e-9))
                 probs[np.isnan(probs)] = 0
            
            probs[probs <= 0] = 0
            total_prob = probs.sum()
            if total_prob > 0:
                probs = probs / total_prob
            else:
                valid_indices = np.where(absent_mask.ravel() > 0)[0]
                probs = np.zeros_like(sim_flat)
                if len(valid_indices) > 0:
                    probs[valid_indices] = 1.0 / len(valid_indices)

            n_add = min(self.add_n_edges, int(np.sum(absent_mask > 0)))
            if n_add > 0 and probs.sum() > 0:
                chosen = np.random.choice(len(probs), size=n_add, replace=False, p=probs)
                i_idx, j_idx = np.unravel_index(chosen, (self.num_nodes, self.num_nodes))
                adj_proposal[i_idx, j_idx] = 1
                
        else:
            existing_edges = np.array(np.where(adj_proposal > 0)).T
            if existing_edges.shape[0] > 0:
                n_remove = min(self.remove_n_edges, existing_edges.shape[0])
                remove_idx = np.random.choice(existing_edges.shape[0], size=n_remove, replace=False)
                to_remove = existing_edges[remove_idx]
                adj_proposal[to_remove[:, 0], to_remove[:, 1]] = 0
                
        return adj_proposal

    def refine_adj_epoch(self, current_acc):
        """
        Main method to be called once per training epoch.
        Runs an isothermal simulation and then updates the temperature adaptively.
        """
        if self.time_step >= self.t_max:
            return self._make_sparse_adj(self.adj_sub)

        self.energy_history = []

        for _ in range(self.isothermal_steps):
            adj_proposal = self._propose_move(current_acc)
            energy_proposal, _ = self._evaluate_energy(adj_proposal, current_acc)
            
            if self.prev_energy < -1e11:
                self.prev_energy, _ = self._evaluate_energy(self.adj_sub, current_acc)

            delta_e = energy_proposal - self.prev_energy
            
            if delta_e >= 0 or np.random.rand() < np.exp(delta_e / (self.temperature + 1e-9)):
                self.adj_sub = adj_proposal
                self.prev_energy = energy_proposal
            
            self.energy_history.append(self.prev_energy)

        if len(self.energy_history) > 1:
            energy_variance = np.var(self.energy_history)
            
            damping_factor = np.tanh(energy_variance / (self.initial_temp + 1e-9))
            adaptive_rate = self.base_adaptive_cooling_rate + (1 - self.base_adaptive_cooling_rate) * damping_factor
            self.temperature *= adaptive_rate
        else:
            self.temperature *= self.base_adaptive_cooling_rate
            
        self.temperature = max(self.final_temp, self.temperature)

        self.prev_acc = current_acc
        self.time_step += 1

        return self._make_sparse_adj(self.adj_sub)

    def _make_sparse_adj(self, adj_sub):
        edge_index = torch.tensor(adj_sub, dtype=torch.float32).nonzero(as_tuple=False).t().contiguous()
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=self.num_nodes)
        adj_sparse = torch.sparse_coo_tensor(
            edge_index, torch.ones(edge_index.size(1)), (self.num_nodes, self.num_nodes)
        ).to(self.device)
        return adj_sparse

    @staticmethod
    def _dcg_at_k(relevance, k):
        relevance = np.asfarray(relevance)[:, :k]
        discounts = np.log2(np.arange(1, k + 1) + 1)
        return np.sum(relevance / discounts, axis=1)

    @staticmethod
    def _ndcg_at_k(predictions, ground_truth, k):
        ranked_indices = np.argsort(predictions, axis=1)[:, ::-1]
        relevance = np.take_along_axis(ground_truth, ranked_indices, axis=1)
        dcg = PhysicsRefiner._dcg_at_k(relevance, k)
        ideal_relevance = np.sort(ground_truth, axis=1)[:, ::-1]
        idcg = PhysicsRefiner._dcg_at_k(ideal_relevance, k)
        return np.where(idcg > 0, dcg / idcg, 0.0)