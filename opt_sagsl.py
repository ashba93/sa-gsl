import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from copy import deepcopy
from torch_geometric.utils import add_self_loops, remove_self_loops, to_scipy_sparse_matrix, homophily
from torch_geometric.nn.models import LabelPropagation
from torch_geometric.utils import dropout_adj
import networkx as nx
import scipy.sparse

from opengsl.data.dataset import Dataset
from opengsl.module.encoder import GNNEncoder_OpenGSL
from opengsl.utils import set_seed, accuracy
from opengsl.module.functional import normalize
from SARefiner import SimulatedAnnealingRefiner

from optuna.trial import TrialState
import argparse
import optuna
import json
import time

def analyze_graph_properties(adj, labels, graph_name="Graph"):
    """
    Computes and prints structural properties of a given graph.

    Args:
        adj (torch.Tensor): The adjacency matrix (can be sparse or dense).
        labels (torch.Tensor): The node labels.
        graph_name (str): A name for the graph being analyzed (e.g., "Original" or "Learned").
    """
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
        scipy_adj = to_scipy_sparse_matrix(edge_index, num_nodes=adj.shape[0])
    else:
        adj_binary = (adj > 0).float()
        edge_index = adj_binary.nonzero(as_tuple=False).t()
        scipy_adj = to_scipy_sparse_matrix(edge_index, num_nodes=adj.shape[0])

    G = nx.from_scipy_sparse_array(scipy_adj)
    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    density = nx.density(G)
    
    avg_degree = num_edges * 2 / num_nodes
    
    homophily_ratio = homophily(edge_index.to(labels.device), labels)
    
    properties = {
        "num_edges": num_edges,
        "density": density,
        "avg_degree": avg_degree,
        "homophily": homophily_ratio
    }

    print(f"\n--- Properties for {graph_name} ---")
    for key, value in properties.items():
        print(f"  {key:<20}: {value:.4f}")
    
    return properties

def drop_edges(A, rate=0.2):
    idx = torch.nonzero(A.to_dense(), as_tuple=False)
    mask = torch.rand(idx.shape[0]) > rate
    new_idx = idx[mask]
    return torch.sparse_coo_tensor(new_idx.t().to(A.device), torch.ones(new_idx.shape[0]).to(A.device), A.shape).to(A.device)


def ensemble_sa_adjs(sa, n_graphs=3, dropedge_rate=0.1, prev_train_acc=0):
    """Generate multiple SA-refined adjacencies and apply DropEdge"""
    adjs = []
    for _ in range(n_graphs):
        adj_sparse, energy = sa.step(current_acc=prev_train_acc)
        adj_dropped = drop_edges(adj_sparse, rate=dropedge_rate)
        adjs.append(adj_dropped.to_dense())
    # average ensemble
    stacked = torch.stack(adjs, dim=0)
    A_ens_dense = (stacked.sum(dim=0) >= (n_graphs // 2 + 1)).float()
    return A_ens_dense.to_sparse(), energy

def train(dataset,
          alpha,
          beta,
          gamma,
          lambda_,
          delta,
          max_allowed_degree,
          add_fraction,
          alpha_start,
          alpha_end,
          dropedge_rate,
          n_ensemble,
          lr,
          weight_decay,
          reg_weight,
          n_hidden,
          n_layers,
          dropout,
          seed):
    train_mask = dataset.train_masks[0]
    val_mask = dataset.val_masks[0]
    test_mask = dataset.test_masks[0]
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    print(device)
    set_seed(seed)

    num_nodes = int(dataset.adj.shape[0])
    num_classes = int(dataset.n_classes)

    train_nodes = np.where(train_mask)[0]
    sa_train_nodes, sa_inner_val_nodes = train_test_split(train_nodes, test_size=0.4, random_state=42)

    sa_train_mask = np.zeros_like(train_mask)
    sa_train_mask[sa_train_nodes] = True
    sa_inner_val_mask = np.zeros_like(train_mask)
    sa_inner_val_mask[sa_inner_val_nodes] = True

    l_sa_train = np.zeros((num_nodes, num_classes), dtype=float)
    l_sa_train[sa_train_mask, dataset.labels[sa_train_mask]] = 1.0
    l_sa_val = np.zeros((num_nodes, num_classes), dtype=float)
    l_sa_val[sa_inner_val_mask, dataset.labels[sa_inner_val_mask]] = 1.0

    # check homophily
    if analyze_graph_properties(adj=dataset.adj, labels=dataset.labels, graph_name='Original Graph')["homophily"] >= 0.5:
        homophily_check = True
    else:
        homophily_check = False

    n_epochs = 1000

    sa = SimulatedAnnealingRefiner(
        num_nodes=num_nodes,
        feats=dataset.feats.cpu().numpy(),
        l_train=l_sa_train,
        l_val=l_sa_val,
        t_max=n_epochs,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        lambda_=lambda_,
        delta=delta,
        max_allowed_degree=max_allowed_degree,
        add_fraction=add_fraction,
        device=device
    )

    gnn = GNNEncoder_OpenGSL(
        dataset.dim_feats,
        n_hidden=n_hidden,
        n_class=num_classes,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)


    best_val_acc = 0.0
    best_weights = None
    prev_train_acc = 0.0 
    alpha_start = alpha_start
    alpha_end = alpha_end
    dropedge_rate = dropedge_rate
    n_ensemble = n_ensemble
    epochs_no_improve = 0
    patience = 100

    test_acc_l, energy_l = [], []

    optim = torch.optim.Adam(list(gnn.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        optim.zero_grad()

        A_ref_ens, energy = ensemble_sa_adjs(sa=sa, n_graphs=n_ensemble, dropedge_rate=dropedge_rate, prev_train_acc=prev_train_acc)

        A_ref_ens = A_ref_ens.to_dense().to(device)

        A_raw = dataset.adj.to(device)

        A_raw = A_raw.to_dense()

        if homophily_check:
            alpha = alpha_start - (alpha_start - alpha_end) * (epoch / n_epochs)
            A_ens = alpha * A_raw + (1 - alpha) * A_ref_ens
        else:
            A_ens = A_ref_ens

        output = gnn(dataset.feats.to(device), A_ens.to(device))

        adj_density = (A_ref_ens != 0).float().mean()
        degree_penalty = torch.clamp(A_ref_ens.sum(1) - sa.max_allowed_degree, min=0).sum() / num_nodes

        reg_term = reg_weight * adj_density + reg_weight * degree_penalty

        loss_train = F.cross_entropy(output[train_mask].to(device), dataset.labels[train_mask].to(device)) + reg_term
        loss_train.backward()
        optim.step()
        scheduler.step()

        train_acc = accuracy(dataset.labels[train_mask].cpu().numpy(),
                             output[train_mask].detach().cpu().numpy())

        gnn.eval()
        with torch.no_grad():
            out_val = gnn(dataset.feats.to(device), A_ens.to(device))
            loss_val = F.cross_entropy(out_val[val_mask].to(device), dataset.labels[val_mask].to(device))
            val_acc = accuracy(dataset.labels[val_mask].cpu().numpy(),
                               out_val[val_mask].detach().cpu().numpy())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(gnn.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            test_acc_l.append(accuracy(dataset.labels[test_mask].cpu().numpy(), out_val[test_mask].detach().cpu().numpy()))
            energy_l.append(energy)
            print(f"Epoch {epoch:03d} | Train loss {loss_train.item():.4f} | Train acc {train_acc:.4f} | Val loss {loss_val.item():.4f} | Val acc {val_acc:.4f} | Energy {energy:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        prev_train_acc = train_acc

    # final test
    if best_weights is not None:
        gnn.load_state_dict(best_weights)

    gnn.eval()
    with torch.no_grad():
        out_test = gnn(dataset.feats.to(device), A_ens.to(device))
        loss_test = F.cross_entropy(out_test[test_mask].to(device), dataset.labels[test_mask].to(device))
        test_acc = accuracy(dataset.labels[test_mask].cpu().numpy(),
                            out_test[test_mask].detach().cpu().numpy())

    print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {loss_test.item():.4f}")

    return best_val_acc, test_acc, A_ens.detach().cpu(), test_acc_l, energy_l, out_test.detach().cpu()

def objective(trial):
    alpha= trial.suggest_float("alpha",0.1,100.0)
    beta= trial.suggest_float("beta",0.1,100.0)
    gamma= trial.suggest_float("gamma",0.1,100.0)
    lambda_= trial.suggest_float("lambda_",0.1,100.0)
    delta= trial.suggest_float("delta",0.1,100.0)
    max_allowed_degree= trial.suggest_int("max_allowed_degree",3,50)
    add_fraction=trial.suggest_float("add_fraction",0.1,0.3)
    alpha_start = trial.suggest_float("alpha_start",0.7,0.99)
    alpha_end = trial.suggest_float("alpha_end",0.01,0.7)
    dropedge_rate = trial.suggest_float("dropedge_rate",0.01,0.1)
    n_ensemble = trial.suggest_int("n_ensemble",1,7)
    lr=trial.suggest_float("lr",5e-7,5e-1)
    weight_decay=trial.suggest_float("weight_decay",1e-5,1e-1)
    reg_weight=trial.suggest_float("reg_weight",5e-5,5e-1)
    n_hidden = trial.suggest_int("n_hidden",4,64)
    n_layers = trial.suggest_int("n_layers",1,1)
    dropout = trial.suggest_float("dropout",0,1)

    seed=42

    best_val_acc, _, learned_adj, __, ___, final_embeddings  = train(dataset,
                                                                   alpha,
                                                                   beta,
                                                                   gamma,
                                                                   lambda_,
                                                                   delta,
                                                                   max_allowed_degree,
                                                                   add_fraction,
                                                                   alpha_start,
                                                                   alpha_end,
                                                                   dropedge_rate,
                                                                   n_ensemble,
                                                                   lr,
                                                                   weight_decay,
                                                                   reg_weight,
                                                                   n_hidden,
                                                                   n_layers,
                                                                   dropout,
                                                                   seed)

    return best_val_acc

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

    print("\n" + "="*20 + " Analyzing Original Graph " + "="*20)
    original_properties = analyze_graph_properties(dataset.adj, dataset.labels, "Original Graph")

    results = []
    learned_graph_properties_list = []

    for seed in range(10):
        print(f"\n=== Run {seed+1}/10 with seed {seed} ===")
        start_time = time.perf_counter()
        val_acc, test_acc, learned_adj, test_acc_l, energy_l, final_embeddings = train(dataset,
                                                                                        best_params["alpha"],
                                                                                        best_params["beta"],
                                                                                        best_params["gamma"],
                                                                                        best_params["lambda_"],
                                                                                        best_params["delta"],
                                                                                        best_params["max_allowed_degree"],
                                                                                        best_params["add_fraction"],
                                                                                        best_params["alpha_start"],
                                                                                        best_params["alpha_end"],
                                                                                        best_params["dropedge_rate"],
                                                                                        best_params["n_ensemble"],
                                                                                        best_params["lr"],
                                                                                        best_params["weight_decay"],
                                                                                        best_params["reg_weight"],
                                                                                        best_params["n_hidden"],
                                                                                        best_params["n_layers"],
                                                                                        best_params["dropout"],
                                                                                        seed=seed)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        results.append(test_acc)

        print(f"Execution Time: {execution_time:.6f}")
        print(f"Test accuracy: {test_acc_l}")
        print(f"Energy: {energy_l}")

        if seed == 0:
            print("\nSaving outputs from the first run for t-SNE visualization...")
            torch.save(final_embeddings, f'{dataset_name}_embeddings.pt')
            torch.save(learned_adj, f'{dataset_name}_learned_adj.pt')
            torch.save(dataset.labels, f'{dataset_name}_labels.pt')
            
        run_properties = analyze_graph_properties(learned_adj, dataset.labels, f"Learned Graph (Run {seed+1})")
        learned_graph_properties_list.append(run_properties)

    print("\n=== Final Results over 10 runs ===")
    print("Accuracies:", results)
    print("Mean:", np.mean(results))
    print("Std:", np.std(results))

    print("\n" + "="*20 + " RQ2: Structural Properties Comparison " + "="*20)
    
    avg_learned_props = {}
    std_learned_props = {}
    property_keys = learned_graph_properties_list[0].keys()
    
    for key in property_keys:
        values = [props[key] for props in learned_graph_properties_list]
        avg_learned_props[key] = np.mean(values)
        std_learned_props[key] = np.std(values)

    print(f"{'Property':<20} | {'Original Graph':<20} | {'Learned Graph (Avg ± Std)':<30}")
    print("-" * 75)
    for key in property_keys:
        original_val = f"{original_properties[key]:.4f}"
        learned_val = f"{avg_learned_props[key]:.4f} ± {std_learned_props[key]:.4f}"
        print(f"{key:<20} | {original_val:<20} | {learned_val:<30}")