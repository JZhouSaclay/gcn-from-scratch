"""
t-SNE Visualization of GCN Hidden Features
==========================================
Visualize the node embeddings learned by the first GCN layer.

This script:
1. Trains a GCN model on citation network dataset
2. Extracts the hidden layer features
3. Uses t-SNE to project them to 2D
4. Plots the results with different colors for each class

Usage:
    python visualize.py --dataset cora
    python visualize.py --dataset citeseer

Output:
    results/{dataset}_tsne.png
"""

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from src.data_loader import CoraDataLoader, CiteSeerDataLoader
from src.model import GCN
from src.train import train_gcn
from src.utils import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='t-SNE Visualization of GCN Hidden Features'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='cora',
        choices=['cora', 'citeseer'],
        help='Dataset to use (default: cora)'
    )

    parser.add_argument(
        '--hidden',
        type=int,
        default=None,
        help='Number of hidden units (default: 16 for cora, 32 for citeseer)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs (default: 200)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=5e-4,
        help='L2 regularization (default: 5e-4)'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate (default: 0.5)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (default: 50)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    return parser.parse_args()


def extract_hidden_features(model, x, adj):
    """
    Extract hidden layer features (after first GCN layer and ReLU).

    Args:
        model: Trained GCN model
        x: Node features [N, F]
        adj: Normalized adjacency matrix [N, N]

    Returns:
        hidden_features: Hidden layer activations [N, H] (H=16)
    """
    model.eval()
    with torch.no_grad():
        # First GCN layer
        h = model.gc1(x, adj)
        # ReLU activation
        h = F.relu(h)
        # Note: We don't apply dropout here for stable visualization
    return h.cpu().numpy()


def visualize_tsne(features, labels, class_names, save_path='results/cora_tsne.png'):
    """
    Visualize node embeddings using t-SNE.

    Args:
        features: Hidden features [N, H] (typically 16-dim)
        labels: Node labels [N]
        class_names: List of class names
        save_path: Path to save the figure
    """
    print("\nApplying t-SNE dimensionality reduction...")
    print(f"  Input shape: {features.shape}")

    # Apply t-SNE to reduce to 2D
    # perplexity should be less than number of samples
    # For Cora with 2708 nodes, perplexity=30 is reasonable
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate='auto',
        init='pca',
        random_state=42,
        max_iter=1000,
        verbose=0
    )

    features_2d = tsne.fit_transform(features)
    print(f"  Output shape: {features_2d.shape}")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define 7 high-contrast colors for the 7 classes
    # Using a carefully selected color palette for maximum distinguishability
    colors = [
        '#E63946',  # Red - Case_Based
        '#F4A261',  # Orange - Genetic_Algorithms
        '#2A9D8F',  # Teal - Neural_Networks
        '#264653',  # Dark Blue - Probabilistic_Methods
        '#9B5DE5',  # Purple - Reinforcement_Learning
        '#00BBF9',  # Light Blue - Rule_Learning
        '#F15BB5',  # Pink - Theory
    ]

    num_classes = len(class_names)

    # Plot each class separately for legend
    for i in range(num_classes):
        # Get indices of nodes belonging to class i
        mask = (labels == i)
        class_features = features_2d[mask]

        # Plot with some transparency for overlapping points
        ax.scatter(
            class_features[:, 0],
            class_features[:, 1],
            c=colors[i],
            label=class_names[i],
            alpha=0.7,
            s=50,  # marker size
            edgecolors='white',
            linewidth=0.5
        )

    # Customize the plot
    ax.set_title(
        't-SNE Visualization of GCN Hidden Features (Cora)',
        fontsize=18,
        fontweight='bold',
        pad=20
    )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)

    # Add legend with custom positioning
    legend = ax.legend(
        loc='best',  # Automatically find best position
        fontsize=11,
        frameon=True,
        fancybox=True,
        shadow=True,
        title='Paper Categories',
        title_fontsize=12
    )

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.3)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Tight layout
    plt.tight_layout()

    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Visualization saved to: {save_path}")

    # Also show the plot
    plt.show()

    return features_2d


def analyze_clustering(features_2d, labels, class_names):
    """
    Analyze how well-separated the clusters are in the t-SNE visualization.

    Args:
        features_2d: 2D t-SNE features [N, 2]
        labels: Node labels [N]
        class_names: List of class names
    """
    print("\n" + "="*60)
    print("                    Clustering Analysis")
    print("="*60)

    num_classes = len(class_names)

    # Calculate cluster centers
    centers = []
    for i in range(num_classes):
        mask = (labels == i)
        center = features_2d[mask].mean(axis=0)
        centers.append(center)
    centers = np.array(centers)

    # Calculate intra-cluster distances (compactness)
    print("\nIntra-cluster Compactness (lower = more compact):")
    print("-" * 60)
    intra_dists = []
    for i in range(num_classes):
        mask = (labels == i)
        points = features_2d[mask]
        center = centers[i]
        # Average distance to center
        dists = np.sqrt(np.sum((points - center) ** 2, axis=1))
        avg_dist = dists.mean()
        intra_dists.append(avg_dist)
        print(f"  {class_names[i]:<25} {avg_dist:.3f}")

    avg_intra = np.mean(intra_dists)
    print(f"  {'Average':<25} {avg_intra:.3f}")

    # Calculate inter-cluster distances (separation)
    print("\nInter-cluster Separation (higher = more separated):")
    print("-" * 60)

    from itertools import combinations

    inter_dists = []
    for i, j in combinations(range(num_classes), 2):
        dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
        inter_dists.append(dist)

    avg_inter = np.mean(inter_dists)
    min_inter = np.min(inter_dists)

    print(f"  Average distance between centers: {avg_inter:.3f}")
    print(f"  Minimum distance between centers: {min_inter:.3f}")

    # Silhouette-like score
    print(f"\nClustering Quality Score:")
    print(f"  Separation/Compactness ratio: {avg_inter/avg_intra:.3f}")
    print(f"  (Higher values indicate better clustering)")

    print("\n" + "="*60)


def main():
    """Main execution function for visualization."""
    args = parse_arguments()

    print("\n" + "="*70)
    print("         t-SNE Visualization of GCN Hidden Features")
    print("="*70)
    print()
    print("This script will:")
    print(f"  1. Train a GCN model on {args.dataset.upper()} dataset")
    print("  2. Extract hidden layer features")
    print("  3. Apply t-SNE to reduce to 2D")
    print("  4. Visualize the node embeddings colored by class")
    print()

    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = args.seed
    EPOCHS = args.epochs
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    PATIENCE = args.patience
    DROPOUT = args.dropout

    # Set hidden size based on dataset if not provided
    if args.hidden is None:
        HIDDEN = 32 if args.dataset == 'citeseer' else 16
    else:
        HIDDEN = args.hidden

    # Dynamic save path
    SAVE_PATH = f'results/{args.dataset}_tsne.png'

    # Set random seed
    set_seed(SEED)

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == 'cora':
        loader = CoraDataLoader()
    elif args.dataset == 'citeseer':
        loader = CiteSeerDataLoader()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    data = loader.load()

    # Move data to device
    x = data['x'].to(DEVICE)
    adj = data['adj'].to(DEVICE)
    labels = data['y'].to(DEVICE)
    train_mask = data['train_mask'].to(DEVICE)
    val_mask = data['val_mask'].to(DEVICE)
    test_mask = data['test_mask'].to(DEVICE)

    # Get class names (map to more readable format if needed)
    raw_class_names = data.get('class_names', [f"Class {i}" for i in range(data['num_classes'])])

    # Map to more readable names
    class_name_map = {
        'Case_Based': 'Case Based',
        'Genetic_Algorithms': 'Genetic Algorithms',
        'Neural_Networks': 'Neural Networks',
        'Probabilistic_Methods': 'Probabilistic Methods',
        'Reinforcement_Learning': 'Reinforcement Learning',
        'Rule_Learning': 'Rule Learning',
        'Theory': 'Theory'
    }

    class_names = [class_name_map.get(name, name) for name in raw_class_names]

    print(f"Dataset loaded: {data['num_nodes']} nodes, {data['num_features']} features")
    print(f"Classes: {', '.join(class_names)}")
    print()

    # Create and train GCN model
    print("Training GCN model...")
    model = GCN(
        nfeat=data['num_features'],
        nhid=HIDDEN,
        nclass=data['num_classes'],
        dropout=DROPOUT
    ).to(DEVICE)

    results = train_gcn(
        model=model,
        x=x,
        adj=adj,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        verbose=True
    )

    print(f"\nTraining complete! Test accuracy: {results['test_acc']*100:.2f}%")

    # Extract hidden features
    print("\nExtracting hidden layer features...")
    hidden_features = extract_hidden_features(model, x, adj)

    # Apply t-SNE and visualize
    features_2d = visualize_tsne(
        features=hidden_features,
        labels=labels.cpu().numpy(),
        class_names=class_names,
        save_path=SAVE_PATH
    )

    # Analyze clustering quality
    analyze_clustering(features_2d, labels.cpu().numpy(), class_names)

    print("\n" + "="*70)
    print("Visualization complete!")
    print(f"Saved to: {SAVE_PATH}")
    print("="*70)


if __name__ == '__main__':
    main()
