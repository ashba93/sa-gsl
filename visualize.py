import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

def plot_all_visualizations(dataset_name):
    """
    Loads saved embeddings, labels, and the learned graph.
    Plots t-SNE, PCA, and a static graph visualization with edges.
    """
    try:
        embeddings = torch.load(f'{dataset_name}_embeddings.pt').cpu()
        labels = torch.load(f'{dataset_name}_labels.pt').cpu().numpy()
        learned_adj = torch.load(f'{dataset_name}_learned_adj.pt').cpu()
    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure you have run the training script successfully and saved the .pt files.")
        return

    print("Data loaded. Running dimensionality reduction...")

    print("Fitting t-SNE (this might take a while for large datasets)...")
    tsne = TSNE(n_components=2, perplexity=40, n_iter=500, random_state=42)
    tsne_results = tsne.fit_transform(embeddings.numpy())

    print("Generating plots...")
    
    cmap = plt.cm.get_cmap("jet", np.max(labels) + 1)

    # --- Plot 2: t-SNE Visualization ---
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap=cmap, alpha=0.5, s=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(f'tsne_{dataset_name}.pdf', dpi=1000, bbox_inches='tight')
    print(f"t-SNE plot saved to tsne_{dataset_name}.pdf")
    
    plt.show() # Display all plots

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding and Graph Visualization Script")
    parser.add_argument("--dataset", required=True, help="Name of the dataset (e.g., cora, citeseer)")
    args = parser.parse_args()
    
    plot_all_visualizations(args.dataset)