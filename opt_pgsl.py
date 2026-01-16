import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.utils import add_self_loops, remove_self_loops, dropout_adj, to_undirected, k_hop_subgraph
from torch_geometric.nn.models import LabelPropagation

from opengsl.data.dataset import Dataset
from opengsl.module.encoder import GNNEncoder_OpenGSL
from opengsl.utils import set_seed, accuracy
from opengsl.module.functional import normalize
from GraphRefiner import GraphRefiner

from optuna.trial import TrialState
import argparse
import optuna
import json

def accuracy(logits, labels):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if logits.ndim > 1:
        preds = logits.argmax(1)
    else:
        preds = logits
    return np.sum(preds == labels) / len(labels)

def to_sparse_adj(edge_index, edge_weight, num_nodes):
    """
    Converts an edge_index and edge_weights into a sparse
    adjacency matrix tensor.
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
        
    size = (num_nodes, num_nodes)
    
    # Use torch.sparse_coo_tensor for modern PyTorch versions
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, size=size)
    
    return adj

def create_optimizer_and_scheduler(gsl_module, gnn, lr, weight_decay, n_epochs):
    opt = torch.optim.Adam(
        list(gsl_module.parameters()) + list(gnn.parameters()), 
        lr=lr, 
        weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    return opt, sched

def train(dataset,
          n_epochs,
          lr,
          weight_decay,
          n_hidden,
          n_layers,
          dropout,
          edges_per_node,
          alpha,
          seed):

    set_seed(seed)

    train_mask = dataset.train_masks[0]
    val_mask = dataset.val_masks[0]
    test_mask = dataset.test_masks[0]
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    num_nodes = int(dataset.adj.shape[0])
    num_classes = int(dataset.n_classes)

    train_nodes = np.where(train_mask)[0]
    val_nodes = np.where(val_mask)[0]
    test_nodes = np.where(test_mask)[0]

    class GSLModule(nn.Module):
        """Learns an importance score for each potential edge."""
        def __init__(self, input_dim, hidden_dim=64):
            super(GSLModule, self).__init__()
            self.mlp = nn.Sequential(
                nn.Linear(input_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        def forward(self, x, edge_index):
            edge_features = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1)
            return self.mlp(edge_features).squeeze()

    gnn = GNNEncoder_OpenGSL(
        dataset.feats.shape[1],
        n_hidden=n_hidden,
        n_class=num_classes,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)

    refiner = GraphRefiner(dataset,
                           num_nodes)

    n_epochs = n_epochs
    REFINE_EPOCHS = 20
    best_val_acc = 1e-5
    best_weights = None
    prev_train_acc = 1e-5
    epochs_no_improve = 0
    patience = 20

    original_edge_index = refiner._generate_edge_index()
    candidate_non_edges = refiner._generate_structural_candidates()
    
    initial_edge_pool = torch.cat([original_edge_index, candidate_non_edges], dim=1)
    initial_edge_pool = to_undirected(initial_edge_pool)
    
    print(f"\nOriginal Edges: {original_edge_index.shape[1]}")
    print(f"Initial Pool Size (Original + Structural Candidates): {initial_edge_pool.shape[1]}")

    gsl_module = GSLModule(dataset.feats.shape[1] , 64).to(device)
    optimizer = torch.optim.Adam(list(gsl_module.parameters()) + list(gnn.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    active_edge_index = initial_edge_pool
    sparse_orig_adj = dataset.adj

    for epoch in range(1, n_epochs + 1):
        if epoch > 1 and epoch % REFINE_EPOCHS == 0:
            refiner.refine(gsl_module, num_probes=1000, edges_per_node=edges_per_node)
            active_edge_index = refiner.current_graph.to(device)
            # optimizer, scheduler = create_optimizer_and_scheduler(gsl_module, gnn, lr, weight_decay, n_epochs)
            # reset epoch without improve; it needs epochs to re-stabilize
            epochs_no_improve = 2 # allow the patience to stop into a same new refinement
        
        gsl_module.train(); gnn.train()
        optimizer.zero_grad()

        edge_logits = gsl_module(dataset.feats, active_edge_index)
        edge_weights = torch.sigmoid(edge_logits)
        sparse_adj = to_sparse_adj(active_edge_index, edge_weights, num_nodes)

        A_ens = alpha * sparse_orig_adj + (1 - alpha) * sparse_adj

        out = gnn(dataset.feats, A_ens.to_dense())
        loss = F.cross_entropy(out[train_mask], dataset.labels[train_mask])
        
        loss.backward()
        optimizer.step()

        # preds = out.argmax(dim=1)
        train_acc = accuracy(out[train_mask], dataset.labels[train_mask])
        prev_train_acc = train_acc

        gnn.eval(); gsl_module.eval()
        with torch.no_grad():
            eval_logits = gsl_module(dataset.feats, active_edge_index)
            eval_weights = torch.sigmoid(eval_logits)
            # eval_weights = (eval_logits > 0).float()
            val_sparse_adj = to_sparse_adj(active_edge_index, eval_weights, num_nodes)

            A_ens = alpha * sparse_orig_adj + (1 - alpha) * val_sparse_adj

            out_val = gnn(dataset.feats, A_ens.to_dense())
            
            # preds = out.argmax(dim=1)
            val_acc = accuracy(out_val[val_mask], dataset.labels[val_mask])
            loss_val = F.cross_entropy(out_val[val_mask], dataset.labels[val_mask])

            prev_train_acc = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(gnn.state_dict())
            best_active_edges = deepcopy(active_edge_index)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch:03d} | Train loss {loss.item():.4f} | Train acc {train_acc:.4f} | Val loss {loss_val.item():.4f} | Val acc {val_acc:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        prev_train_acc = train_acc

    if best_weights is not None:
        gnn.load_state_dict(best_weights)

    gnn.eval(); gsl_module.eval()
    with torch.no_grad():
        eval_logits = gsl_module(dataset.feats, best_active_edges)
        eval_weights = torch.sigmoid(eval_logits)
        # eval_weights = (eval_logits > 0).float()
        test_sparse_adj = to_sparse_adj(best_active_edges, eval_weights, num_nodes)

        A_ens = alpha * sparse_orig_adj + (1 - alpha) * test_sparse_adj
        
        out_test = gnn(dataset.feats, A_ens.to_dense())
        
        test_acc = accuracy(out_test[test_mask], dataset.labels[test_mask])
        loss_test = F.cross_entropy(out_test[test_mask], dataset.labels[test_mask])
    print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {loss_test.item():.4f}")

    return best_val_acc, test_acc

def objective(trial):
    n_epochs = 200
    lr=trial.suggest_float("lr",5e-7,1e-1)
    weight_decay=trial.suggest_float("weight_decay",1e-6,1e-1)
    n_hidden = trial.suggest_int("n_hidden",4,64)
    n_layers = trial.suggest_int("n_layers",1,3)
    dropout = trial.suggest_float("dropout",0,.999)
    edges_per_node = trial.suggest_int("edges_per_node", 1, 10)
    alpha = trial.suggest_float("alpha", 0.01, 0.99)

    seed=42

    val_acc, _ =train(dataset,
                      n_epochs,
                      lr,
                      weight_decay,
                      n_hidden,
                      n_layers,
                      dropout,
                      edges_per_node,
                      alpha,
                      seed)

    return val_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SA-GSL pipeline")
    parser.add_argument("--dataset", required=True, help="Name of the dataset")

    args = parser.parse_args()
    dataset_name = args.dataset
    dataset = Dataset(dataset_name, n_splits=1, split="random", cv=5)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    best_params = trial.params

    results = []
    for seed in range(10):
        print(f"\n=== Run {seed+1}/10 with seed {seed} ===")
        _, acc = train(dataset,
                        200,
                        best_params["lr"],
                        best_params["weight_decay"],
                        best_params["n_hidden"],
                        best_params["n_layers"],
                        best_params["dropout"],
                        best_params["edges_per_node"],
                        best_params["alpha"],
                        seed=seed)
        results.append(acc)

    print("\n=== Final Results over 10 runs ===")
    print("Accuracies:", results)
    print("Mean:", np.mean(results))
    print("Std:", np.std(results))