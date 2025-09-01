import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.utils import add_self_loops, remove_self_loops
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import dropout_adj

from opengsl.data.dataset import Dataset
from opengsl.module.encoder import GNNEncoder_OpenGSL
from opengsl.utils import set_seed, accuracy
from opengsl.module.functional import normalize
from SARefiner import SimulatedAnnealingRefiner

def drop_edges(A, rate=0.2):
    idx = torch.nonzero(A.to_dense(), as_tuple=False)
    mask = torch.rand(idx.shape[0]) > rate
    new_idx = idx[mask]
    return torch.sparse_coo_tensor(new_idx.t(), torch.ones(new_idx.shape[0]), A.shape).to(A.device)


def ensemble_sa_adjs(sa, n_graphs=3, dropedge_rate=0.1, prev_train_acc=0):
    """Generate multiple SA-refined adjacencies and apply DropEdge"""
    adjs = []
    for _ in range(n_graphs):
        adj_sparse = sa.step(current_acc=prev_train_acc)
        adj_dropped = drop_edges(adj_sparse, rate=dropedge_rate)
        adjs.append(adj_dropped.to_dense())
    # average ensemble
    stacked = torch.stack(adjs, dim=0)
    A_ens_dense = (stacked.sum(dim=0) >= (n_graphs // 2 + 1)).float()
    return A_ens_dense.to_sparse()

def train():
    dataset = Dataset("cora", n_splits=1, split="random", cv=5)#,split_params="train_examples_per_class")
    # dataset = Dataset("wikics")#, n_splits=0)
    train_mask = dataset.train_masks[0]
    val_mask = dataset.val_masks[0]
    test_mask = dataset.test_masks[0]
    device = torch.device("cpu")
    set_seed(42)

    num_nodes = int(dataset.adj.shape[0])
    num_classes = int(dataset.n_classes)

    # inner SA split (from training set only) - no leakage to val/test
    train_nodes = np.where(train_mask)[0]
    sa_train_nodes, sa_inner_val_nodes = train_test_split(train_nodes, test_size=0.4, random_state=42)

    sa_train_mask = np.zeros_like(train_mask)
    sa_train_mask[sa_train_nodes] = True
    sa_inner_val_mask = np.zeros_like(train_mask)
    sa_inner_val_mask[sa_inner_val_nodes] = True

    # label matrices for SA (one-hot on sa_train and sa_inner_val)
    l_sa_train = np.zeros((num_nodes, num_classes), dtype=float)
    l_sa_train[sa_train_mask, dataset.labels[sa_train_mask]] = 1.0
    l_sa_val = np.zeros((num_nodes, num_classes), dtype=float)
    l_sa_val[sa_inner_val_mask, dataset.labels[sa_inner_val_mask]] = 1.0

    # tuned SA hyperparams
    sa = SimulatedAnnealingRefiner(
        num_nodes=num_nodes,
        feats=dataset.feats.cpu().numpy(),
        l_train=l_sa_train,
        l_val=l_sa_val,
        t_max=200,
        alpha=50.0,
        beta=5.0,
        gamma=10.0,
        lambda_=0.1,
        delta=0.1,
        max_allowed_degree=20,
        add_fraction=0.08,
        device=device
    )

    # GNN
    gnn = GNNEncoder_OpenGSL(
        dataset.dim_feats,
        n_hidden=32,
        n_class=num_classes,
        n_layers=2,
        dropout=0.1
    ).to(device)


    n_epochs = 200
    best_val_acc = 0.0
    best_weights = None
    prev_train_acc = 0.0 
    alpha_start = 0.99
    alpha_end = 0.88
    dropedge_rate = 0.01
    n_ensemble = 5
    epochs_no_improve = 0
    patience = 1000

    optim = torch.optim.Adam(list(gnn.parameters()), lr=5e-1, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        optim.zero_grad()

        A_ref_ens = ensemble_sa_adjs(sa=sa, n_graphs=n_ensemble, dropedge_rate=dropedge_rate, prev_train_acc=prev_train_acc)

        A_ref_ens = A_ref_ens.to_dense()

        A_raw = normalize(dataset.adj, add_loop=False).to(device)

        A_raw = A_raw.to_dense()

        alpha = alpha_start - (alpha_start - alpha_end) * (epoch / n_epochs)

        A_ens = alpha * A_raw + (1 - alpha) * A_ref_ens

        output = gnn(dataset.feats, A_ens)

        adj_density = (A_ref_ens != 0).float().mean()
        degree_penalty = torch.clamp(A_ref_ens.sum(1) - sa.max_allowed_degree, min=0).sum() / num_nodes

        reg_term = 5e-2 * adj_density + 5e-2 * degree_penalty

        loss_train = F.cross_entropy(output[train_mask], dataset.labels[train_mask]) + reg_term
        loss_train.backward()
        optim.step()
        scheduler.step()

        train_acc = accuracy(dataset.labels[train_mask].cpu().numpy(),
                             output[train_mask].detach().cpu().numpy())

        gnn.eval()
        with torch.no_grad():
            out_val = gnn(dataset.feats, A_ens)
            loss_val = F.cross_entropy(out_val[val_mask], dataset.labels[val_mask])
            val_acc = accuracy(dataset.labels[val_mask].cpu().numpy(),
                               out_val[val_mask].detach().cpu().numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(gnn.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train loss {loss_train.item():.4f} | Train acc {train_acc:.4f} | Val loss {loss_val.item():.4f} | Val acc {val_acc:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        prev_train_acc = train_acc

    # final test
    if best_weights is not None:
        gnn.load_state_dict(best_weights)

    gnn.eval()
    with torch.no_grad():
        out_test = gnn(dataset.feats, A_ens)
        loss_test = F.cross_entropy(out_test[test_mask], dataset.labels[test_mask])
        test_acc = accuracy(dataset.labels[test_mask].cpu().numpy(),
                            out_test[test_mask].detach().cpu().numpy())

    print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {loss_test.item():.4f}")

if __name__ == "__main__":
    train()