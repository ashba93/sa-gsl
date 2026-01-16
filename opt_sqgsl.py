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
from SQRefiner import PhysicsRefiner

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

def drop_edges(A, rate=0.2):
    idx = torch.nonzero(A.to_dense(), as_tuple=False)
    mask = torch.rand(idx.shape[0]) > rate
    new_idx = idx[mask]
    return torch.sparse_coo_tensor(new_idx.t().to(A.device), torch.ones(new_idx.shape[0]).to(A.device), A.shape).to(A.device)


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

def train(dataset,
          alpha,
          beta,
          delta,
          max_allowed_degree,
          add_fraction,
          n_epochs,
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
          J,
          mu,
          isothermal_steps,
          base_adaptive_cooling_rate,
          initial_temp,
          final_temp,
          lp_layers,
          lp_alpha,
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

    refiner = PhysicsRefiner(
        num_nodes=num_nodes,
        feats=dataset.feats,
        labels=dataset.labels.cpu().numpy(),
        l_train=l_sa_train,
        l_val=l_sa_val,
        t_max=n_epochs,
        alpha=alpha,
        beta=beta,
        delta=delta,
        J=J,           # Increase homophily preference
        mu=-mu,        # Increase sparsity preference
        isothermal_steps=isothermal_steps,
        base_adaptive_cooling_rate=base_adaptive_cooling_rate,
        max_allowed_degree=max_allowed_degree,
        add_fraction=add_fraction,
        initial_temp=initial_temp,
        final_temp=final_temp,
        lp_layers=lp_layers,
        lp_alpha=lp_alpha,
        device=device,
    )



    gnn = GNNEncoder_OpenGSL(
        dataset.dim_feats,
        n_hidden=n_hidden,
        n_class=num_classes,
        n_layers=n_layers,
        dropout=dropout
    ).to(device)


    n_epochs = n_epochs
    best_val_acc = 1e-5
    best_weights = None
    prev_train_acc = 1e-5
    alpha_start = alpha_start
    alpha_end = alpha_end
    dropedge_rate = dropedge_rate
    n_ensemble = n_ensemble
    epochs_no_improve = 0
    patience = 50

    optim = torch.optim.Adam(list(gnn.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=n_epochs)

    for epoch in range(1, n_epochs + 1):
        gnn.train()
        optim.zero_grad()

        A_ref = refiner.refine_adj_epoch(prev_train_acc)
        A_ref = A_ref.to_dense().to(device)

        A_raw = dataset.adj.to_dense().to(device)
        alpha = alpha_start - (alpha_start - alpha_end) * (epoch / n_epochs)
        A_ens = alpha * A_raw + (1 - alpha) * A_ref

        output = gnn(dataset.feats.to(device), A_ens.to(device))

        adj_density = (A_ref != 0).float().mean()
        degree_penalty = torch.clamp(A_ref.sum(1) - refiner.max_allowed_degree, min=0).sum() / num_nodes

        reg_term = reg_weight * adj_density + reg_weight * degree_penalty

        loss_train = F.cross_entropy(output[train_mask].to(device), dataset.labels[train_mask].to(device)) + reg_term
        loss_train.backward()
        optim.step()
        scheduler.step()

        preds = output.argmax(dim=1)
        train_acc = accuracy(preds[train_mask], dataset.labels[train_mask])
        prev_train_acc = train_acc

        gnn.eval()
        with torch.no_grad():
            out_val = gnn(dataset.feats.to(device), A_ens.to(device))
            loss_val = F.cross_entropy(out_val[val_mask].to(device), dataset.labels[val_mask].to(device))

            val_acc = accuracy(preds[val_mask], dataset.labels[val_mask])
            prev_train_acc = val_acc

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(gnn.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # if epoch % 10 == 0 or epoch == 1:
        #     print(f"Epoch {epoch:03d} | Train loss {loss_train.item():.4f} | Train acc {train_acc:.4f} | Val loss {loss_val.item():.4f} | Val acc {val_acc:.4f}")
        print(f"Epoch {epoch:03d} | Train loss {loss_train.item():.4f} | Train acc {train_acc:.4f} | Val loss {loss_val.item():.4f} | Val acc {val_acc:.4f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        prev_train_acc = train_acc

    if best_weights is not None:
        gnn.load_state_dict(best_weights)

    gnn.eval()
    with torch.no_grad():
        out_test = gnn(dataset.feats.to(device), A_ens.to(device))
        loss_test = F.cross_entropy(out_test[test_mask].to(device), dataset.labels[test_mask].to(device))
        test_acc = accuracy(out_test[test_mask].detach().cpu().numpy(),
                            dataset.labels[test_mask].cpu().numpy())
        # test_acc = accuracy(dataset.labels[test_mask].cpu().numpy(),
        #                     out_test[test_mask].detach().cpu().numpy())

    print(f"\nBest Val Acc: {best_val_acc:.4f} | Test Acc: {test_acc:.4f} | Test Loss: {loss_test.item():.4f}")

    return best_val_acc, test_acc

def objective(trial):
    alpha= trial.suggest_float("alpha",0.1,100.0)
    beta= trial.suggest_float("beta",0.1,100.0)
    delta= trial.suggest_float("delta",0.1,100.0)
    max_allowed_degree= trial.suggest_int("max_allowed_degree",3,100)
    add_fraction=trial.suggest_float("add_fraction",0.1,0.3)
    n_epochs = 100
    alpha_start = trial.suggest_float("alpha_start",0.7,0.99)
    alpha_end = trial.suggest_float("alpha_end",0.01,0.7)
    dropedge_rate = trial.suggest_float("dropedge_rate",0.01,0.1)
    n_ensemble = trial.suggest_int("n_ensemble",3,7)
    lr=trial.suggest_float("lr",5e-7,1e-1)
    weight_decay=trial.suggest_float("weight_decay",1e-6,1e-1)
    reg_weight=trial.suggest_float("reg_weight",5e-5,5e-1)
    n_hidden = trial.suggest_int("n_hidden",4,64)
    n_layers = trial.suggest_int("n_layers",1,1)
    dropout = trial.suggest_float("dropout",0,.999)
    J = trial.suggest_float("J", 1, 100.)
    mu = trial.suggest_float("mu", 1, 100.)
    isothermal_steps = trial.suggest_int("isothermal_steps", 1, 3)
    base_adaptive_cooling_rate = trial.suggest_float("base_adaptive_cooling_rate", 0.6, 0.99)
    initial_temp = trial.suggest_float("initial_temp", 0.5, 0.99)
    final_temp = trial.suggest_float("final_temp", 0.01, 0.49)
    lp_layers = trial.suggest_int("lp_layers", 1, 3)
    lp_alpha = trial.suggest_float("lp_alpha", 0.1, 0.9)

    seed=42

    val_acc, _ =train(dataset,
                       alpha,
                       beta,
                       delta,
                       max_allowed_degree,
                       add_fraction,
                       n_epochs,
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
                       J,
                       mu,
                       isothermal_steps,
                       base_adaptive_cooling_rate,
                       initial_temp,
                       final_temp,
                       lp_layers,
                       lp_alpha,
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
                        best_params["alpha"],
                        best_params["beta"],
                        best_params["delta"],
                        best_params["max_allowed_degree"],
                        best_params["add_fraction"],
                        500,
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
                        best_params["J"],
                        best_params["mu"],
                        best_params["isothermal_steps"],
                        best_params["base_adaptive_cooling_rate"],
                        best_params["initial_temp"],
                        best_params["final_temp"],
                        best_params["lp_layers"],
                        best_params["lp_alpha"],
                        seed=seed)
        results.append(acc)

    print("\n=== Final Results over 10 runs ===")
    print("Accuracies:", results)
    print("Mean:", np.mean(results))
    print("Std:", np.std(results))