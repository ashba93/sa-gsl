from opengsl.module.solver import GSLSolver
from opengsl import ExpManager
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import time
from copy import deepcopy

from opengsl.data.dataset import Dataset
from opengsl.module.encoder import GNNEncoder_OpenGSL, GCNDiagEncoder
from opengsl.module import GraphLearner
from opengsl.module.transform import KNN
from opengsl.module.metric import Cosine
from opengsl.module.fuse import Interpolate
from opengsl.utils import set_seed, accuracy
from opengsl.module.functional import normalize

def dcg_at_k(relevance, k):
    relevance = np.asfarray(relevance)[:, :k]
    discounts = np.log2(np.arange(1, k + 1) + 1)
    return np.sum(relevance / discounts, axis=1)

def ndcg_at_k(predictions, ground_truth, k):
    ranked_indices = np.argsort(predictions, axis=1)[:, ::-1]
    relevance = np.take_along_axis(ground_truth, ranked_indices, axis=1)
    dcg = dcg_at_k(relevance, k)
    ideal_relevance = np.sort(ground_truth, axis=1)[:, ::-1]
    idcg = dcg_at_k(ideal_relevance, k)
    return np.where(idcg > 0, dcg / idcg, 0.0)


from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import add_self_loops, remove_self_loops

def structure_learning(
    num_nodes,
    l_train,
    l_test,
    t_max=100,
    alpha=100,
    lambda_=1,
    gamma=1,
    delta=1,
    max_allowed_degree=40,
    device="cpu"
):
    num_edges=num_nodes * num_nodes / t_max / 100
    print(f"Add/Remove Edges:{num_edges}")
    add_n_edges = int(num_edges)
    remove_n_edges = int(num_edges)
    adj_sub = np.zeros((num_nodes, num_nodes), dtype=np.int8)

    prev_energy = -1e12
    earlystop_counter = 0
    time_step = 0

    while time_step <= t_max:
        temperature = 1 / (1 + time_step / t_max)
        adj_sub_copy = adj_sub.copy()

        rnd_x = np.random.randint(0, num_nodes, size=add_n_edges)
        rnd_y = np.random.randint(0, num_nodes, size=add_n_edges)
        adj_sub[rnd_x, rnd_y] = 1

        adj_t = torch.tensor(adj_sub, dtype=torch.float32)
        edge_index = adj_t.nonzero(as_tuple=False).t().contiguous()

        y_tensor = torch.tensor(l_train, dtype=torch.float32).reshape(num_nodes, -1).to(device)
        model = LabelPropagation(num_layers=3, alpha=0.5)
        prop_labels = model(y=y_tensor, edge_index=edge_index.to(device))
        pred_y = prop_labels.squeeze().detach().cpu().numpy()

        ndcg_score = ndcg_at_k(pred_y, l_test, k=5)
        num_edges = np.sum(adj_sub)
        percentage_degree = num_edges / (num_nodes * num_nodes)
        avg_degree = num_edges / adj_sub.shape[0]

        out_degrees = adj_sub.sum(axis=1)
        degree_penalty = np.sum(np.maximum(0, out_degrees - max_allowed_degree))

        alpha = alpha# * (0.5 + 0.5 * np.sin(2 * np.pi * time_step / t_max))
        # gamma+=0.001
        # lambda_+=0.001

        print((np.mean(ndcg_score)),(percentage_degree),
		      lambda_ * (avg_degree),delta * (degree_penalty))

        energy = (
		    alpha * (np.mean(ndcg_score))
		    - gamma * (percentage_degree)
		    - lambda_ * (avg_degree)
		    - delta * (degree_penalty)
		)

        max_degree_in_state = np.max(out_degrees) if num_nodes > 0 else 0

        if energy < prev_energy or percentage_degree > 0.5 or max_degree_in_state >= max_allowed_degree:
            earlystop_counter += 1
            delta_e = energy - prev_energy
            p = min(1, np.exp(-delta_e / temperature))
            if np.random.rand() >= p:
                adj_sub = adj_sub_copy.copy()
                rnd_x = np.random.randint(0, num_nodes, size=remove_n_edges)
                rnd_y = np.random.randint(0, num_nodes, size=remove_n_edges)
                adj_sub[rnd_x, rnd_y] = 0
        else:
            earlystop_counter = 0

        if earlystop_counter == 100:
            break

        prev_energy = energy
        time_step += 1

    print(f"Final output: E={round(energy,3)}, NDCG={round(np.mean(ndcg_score),3)}, Degree={round(percentage_degree*100,3)}%")

    # final edge index
    edge_index = torch.tensor(adj_sub, dtype=torch.float32).nonzero(as_tuple=False).t().contiguous()
    edge_index, _ = remove_self_loops(edge_index)
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    adj_sparse = torch.sparse_coo_tensor(
        edge_index,
        torch.ones(edge_index.size(1)),
        (num_nodes, num_nodes)
    ).to(device)

    return adj_sparse

# ----------------------
# PIPELINE
# ----------------------

# Load dataset
dataset = Dataset("cora", n_splits=1)
train_mask = dataset.train_masks[0]
val_mask = dataset.val_masks[0]
test_mask = dataset.test_masks[0]

device = torch.device("cpu")
set_seed(42)

# --- Build one-hot train/test matrices for label propagation ---
num_nodes = dataset.adj.shape[0]
num_classes = dataset.n_classes



from sklearn.model_selection import train_test_split

train_nodes = np.where(train_mask)[0]
sa_train_nodes, sa_val_nodes = train_test_split(train_nodes, test_size=0.8, random_state=42)

sa_train_mask = np.zeros_like(train_mask)
sa_train_mask[sa_train_nodes] = True

sa_val_mask = np.zeros_like(train_mask)
sa_val_mask[sa_val_nodes] = True

l_sa_train = np.zeros((num_nodes, num_classes), dtype=float)
l_sa_train[sa_train_mask, dataset.labels[sa_train_mask]] = 1.0

l_sa_val = np.zeros((num_nodes, num_classes), dtype=float)
l_sa_val[sa_val_mask, dataset.labels[sa_val_mask]] = 1.0


# l_train_t = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=dataset.labels.device)
# l_train_t[train_mask, dataset.labels[train_mask]] = 1.0

# l_val_t  = torch.zeros((num_nodes, num_classes), dtype=torch.float32, device=dataset.labels.device)
# l_val_t[val_mask,  dataset.labels[val_mask]]  = 1.0

# --- Refine adjacency via structure learning ---
print("Refining graph with simulated annealing + label propagation...")
refined_adj = structure_learning(
    num_nodes=num_nodes,
    l_train=l_sa_train,
    l_test=l_sa_val,
    device=device
)

# --- Build modules ---
# encoder = GCNDiagEncoder(1, dataset.dim_feats)
# metric = Cosine()
# postprocess = [KNN(150)]
# fuse = Interpolate(0.7, 0.3)
# graphlearner = GraphLearner(encoder=encoder, metric=metric, postprocess=postprocess, fuse=fuse).to(device)

gnn = GNNEncoder_OpenGSL(
    dataset.dim_feats,
    n_hidden=8,
    n_class=dataset.n_classes,
    n_layers=1,
    dropout=0.8
).to(device)

# --- Training loop ---
n_epochs = 1000
lr = 1e-5
wd = 1e-4
best_valid = 0
gsl_weights = None
gnn_weights = None
start_time = time.time()

optim = torch.optim.Adam(
    [{'params': gnn.parameters()}],
    lr=lr,
    weight_decay=wd
)

for epoch in range(n_epochs):
    gnn.train()
    # graphlearner.train()
    optim.zero_grad()

    # forward (use refined adjacency instead of raw dataset.adj)
    adj = normalize(refined_adj)#graphlearner(dataset.feats, normalize(refined_adj))
    output = gnn(dataset.feats, normalize(adj, add_loop=False))

    loss_train = F.cross_entropy(output[train_mask], dataset.labels[train_mask])
    acc_train = accuracy(dataset.labels[train_mask].cpu().numpy(), output[train_mask].detach().cpu().numpy())
    loss_train.backward()
    optim.step()

    # validation
    gnn.eval()
    # graphlearner.eval()
    with torch.no_grad():
        adj = normalize(refined_adj)#graphlearner(dataset.feats, normalize(refined_adj))
        output = gnn(dataset.feats, normalize(adj, add_loop=False))
        loss_val = F.cross_entropy(output[val_mask], dataset.labels[val_mask])
        acc_val = accuracy(dataset.labels[val_mask].cpu().numpy(), output[val_mask].detach().cpu().numpy())

    if acc_val > best_valid:
        # gsl_weights = deepcopy(graphlearner.state_dict())
        gnn_weights = deepcopy(gnn.state_dict())
        best_valid = acc_val
        total_time = time.time() - start_time

    print(f"Epoch {epoch+1:05d} | Loss(train) {loss_train.item():.4f} | Acc(train) {acc_train:.4f} | Loss(val) {loss_val:.4f} | Acc(val) {acc_val:.4f}")

print("Optimization Finished!")
print("Total time: {:.4f}s".format(total_time))

# test
# graphlearner.load_state_dict(gsl_weights)
gnn.load_state_dict(gnn_weights)
with torch.no_grad():
    adj = normalize(refined_adj)#graphlearner(dataset.feats, normalize(refined_adj))
    output = gnn(dataset.feats, normalize(adj, add_loop=False))
    loss_test = F.cross_entropy(output[test_mask], dataset.labels[test_mask])
    acc_test = accuracy(dataset.labels[test_mask].cpu().numpy(), output[test_mask].detach().cpu().numpy())

print(f"Loss(test) {loss_test.item():.4f} | Acc(test) {acc_test:.4f}")
