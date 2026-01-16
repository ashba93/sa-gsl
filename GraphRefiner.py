import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected, to_scipy_sparse_matrix, from_scipy_sparse_matrix

class DSU:
    """A Disjoint Set Union (DSU) data structure with path compression and union by size."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.sz = [1] * n
        self.max_size = 1 if n > 0 else 0

    def find(self, i):
        """Finds the representative (root) of the component containing i."""
        if self.parent[i] == i:
            return i
        # Path compression
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        """Merges the components containing i and j."""
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size
            if self.sz[root_i] < self.sz[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.sz[root_i] += self.sz[root_j]
            self.max_size = max(self.max_size, self.sz[root_i])

class GraphRefiner:
    """
    Manages all logic for generating, pruning, and augmenting the graph structure
    using percolation theory.
    """
    def __init__(self, data, num_nodes):
        print("--- Initializing Graph Refiner ---")
        self.data = data
        self.num_nodes = num_nodes
        self.device = data.feats.device
        self.adj = data.adj.to_dense().to(self.device)
        edge_index = self._generate_edge_index()
        candidates = self._generate_structural_candidates()
        
        initial_pool = torch.cat([edge_index, candidates], dim=1)
        self.active_edge_index = to_undirected(initial_pool)
        
        print(f"Original Edges: {edge_index.shape[1]}")
        print(f"Initial Pool Size (Original + Candidates): {self.active_edge_index.shape[1]}")

    @property
    def current_graph(self):
        """Provides easy access to the current graph structure."""
        return self.active_edge_index

    @torch.no_grad()
    def _generate_edge_index(self):
        adj_t = torch.tensor(self.adj, dtype=torch.float32)
        edge_index = adj_t.nonzero(as_tuple=False).t().contiguous()
        return edge_index

    @torch.no_grad()
    def _generate_structural_candidates(self):
        """Generates candidate edges by finding all 2-hop neighbors."""
        print("Generating structural (2-hop) candidate edges...")
        num_nodes = self.num_nodes
        adj_2hop = self.adj @ self.adj
        adj_2hop = adj_2hop.multiply(self.adj == 0).cpu().numpy()
        adj_2hop = sp.csr_matrix(adj_2hop)
        adj_2hop.setdiag(0)
        adj_2hop.eliminate_zeros()
        candidate_edges, _ = from_scipy_sparse_matrix(adj_2hop)
        print(f"Generated {candidate_edges.shape[1]} potential new edges.")
        return candidate_edges.to(self.device)

    @torch.no_grad()
    def refine(self, gsl_model, num_probes=1000, edges_per_node=3):
        """
        Performs a highly efficient refinement using SAMPLING to approximate the
        LCC growth curve, combined with a target sparsity heuristic.
        """
        print("\n--- Starting SAMPLING-BASED Refinement Phase ---")
        gsl_model.eval()
        
        if isinstance(self.data.feats, torch.Tensor):
            feats_tensor = self.data.feats.to(self.device)
        else:
            feats_dense = self.data.feats.todense() if hasattr(self.data.feats, 'todense') else self.data.feats
            feats_tensor = torch.from_numpy(feats_dense).float().to(self.device)
        
        edge_scores = gsl_model(feats_tensor, self.active_edge_index)
        sorted_indices = torch.argsort(edge_scores, descending=True)
        sorted_edges = self.active_edge_index[:, sorted_indices]
        
        edges_np = sorted_edges.cpu().T.numpy()
        num_total_edges = edges_np.shape[0]

        if num_total_edges == 0:
            print("Refinement complete. No edges in the pool.")
            return

        num_to_keep = int(0.8 * num_total_edges)
        num_to_add = num_total_edges - num_to_keep

        probe_indices = np.linspace(0, num_to_add - 1, num=min(num_probes, num_to_add), dtype=int)
        
        # --- Instantiates the standalone DSU class ---
        dsu = DSU(self.num_nodes)
        
        lcc_sizes_at_probes = []
        last_edge_idx = 0
        
        for current_probe_idx in probe_indices:
            edges_to_add = edges_np[last_edge_idx:current_probe_idx + 1]
            for u, v in edges_to_add:
                dsu.union(u, v)
            
            lcc_sizes_at_probes.append(dsu.max_size)
            last_edge_idx = current_probe_idx + 1
            
        growth = np.diff(np.array(lcc_sizes_at_probes), prepend=0)
        critical_probe_index = np.argmax(growth)
        min_edges_for_connectivity = probe_indices[critical_probe_index] + 1

        target_edge_count = self.num_nodes * edges_per_node
        
        num_final_edges = max(min_edges_for_connectivity, target_edge_count)
        num_final_edges = min(num_final_edges, num_total_edges)
        
        print(f"Connectivity (sampled) requires ~{min_edges_for_connectivity} edges. Target is {target_edge_count}.")
        
        self.active_edge_index = sorted_edges[:, :num_final_edges]
        print(f"Refinement complete. New graph has {num_final_edges} edges.")