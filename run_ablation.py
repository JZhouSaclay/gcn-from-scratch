"""
Ablation Study: MLP vs GCN
==========================
Compare the performance of a standard MLP (no graph structure)
vs GCN (with graph convolution) on citation network datasets.

This demonstrates the importance of neighbor aggregation in GCN.
Both models have identical architecture dimensions
and are trained with the same hyperparameters.

Usage:
    python run_ablation.py --dataset cora
    python run_ablation.py --dataset citeseer
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from src.data_loader import CoraDataLoader, CiteSeerDataLoader
from src.model import GCN, MLP
from src.train import train_gcn
from src.utils import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Ablation Study: MLP vs GCN on Citation Networks'
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


def train_model(model_name, model, x, adj, labels, train_mask, val_mask, test_mask,
                epochs=200, lr=0.01, weight_decay=5e-4, patience=50, seed=42):
    """
    Train a model and return the test accuracy.

    Args:
        model_name: Name of the model (for logging)
        model: Model instance (GCN or MLP)
        x: Node features
        adj: Adjacency matrix (ignored by MLP)
        labels: Ground truth labels
        train_mask, val_mask, test_mask: Data split masks
        epochs: Max training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        patience: Early stopping patience
        seed: Random seed

    Returns:
        results: Dictionary with test accuracy and other metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print('='*60)

    # Set seed for reproducibility (same for both models)
    set_seed(seed)

    # Re-initialize model weights with same seed
    def reset_weights(m):
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    model.apply(reset_weights)

    # Train the model
    results = train_gcn(
        model=model,
        x=x,
        adj=adj,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        verbose=True
    )

    return results


def print_comparison_table(gcn_results, mlp_results, dataset_name, nfeat, nclass, nhid):
    """
    Print a beautiful comparison table of GCN vs MLP results.
    """
    print("\n" + "="*70)
    print(f"                    ABLATION STUDY: MLP vs GCN on {dataset_name.upper()}")
    print("="*70)
    print()
    print("Model Architecture Comparison:")
    print("-" * 70)
    print(f"{'Layer':<20} {'MLP':<25} {'GCN':<25}")
    print("-" * 70)
    print(f"{'Layer 1':<20} {f'Linear({nfeat}, {nhid})':<25} {f'GraphConv({nfeat}, {nhid})':<25}")
    print(f"{'Activation':<20} {'ReLU':<25} {'ReLU':<25}")
    print(f"{'Dropout':<20} {'p=0.5':<25} {'p=0.5':<25}")
    print(f"{'Layer 2':<20} {f'Linear({nhid}, {nclass})':<25} {f'GraphConv({nhid}, {nclass})':<25}")
    gcn_params = gcn_results.get('num_params', 'N/A')
    mlp_params = mlp_results.get('num_params', 'N/A')
    if isinstance(gcn_params, int):
        print(f"{'Parameters':<20} {mlp_params:>12,}{'':<12} {gcn_params:>12,}{'':<12}")
    else:
        print(f"{'Parameters':<20} {str(mlp_params):<25} {str(gcn_params):<25}")
    print("-" * 70)
    print()
    print("Key Difference:")
    print("  • MLP: Each node is classified INDEPENDENTLY (no neighbor info)")
    print("  • GCN: Each node AGGREGATES features from its neighbors")
    print()
    print("="*70)
    print("                         RESULTS SUMMARY")
    print("="*70)
    print()

    # Results table
    gcn_test_acc = gcn_results['test_acc'] * 100
    mlp_test_acc = mlp_results['test_acc'] * 100
    gap = gcn_test_acc - mlp_test_acc

    gcn_train_acc = gcn_results['train_acc'] * 100
    mlp_train_acc = mlp_results['train_acc'] * 100

    gcn_val_acc = gcn_results['val_acc'] * 100
    mlp_val_acc = mlp_results['val_acc'] * 100

    gcn_epoch = gcn_results['best_epoch']
    mlp_epoch = mlp_results['best_epoch']

    gcn_time = gcn_results['training_time']
    mlp_time = mlp_results['training_time']

    print(f"{'Metric':<30} {'MLP':>15} {'GCN':>15} {'Improvement':>15}")
    print("-" * 70)
    print(f"{'Train Accuracy':<30} {mlp_train_acc:>14.2f}% {gcn_train_acc:>14.2f}% {'N/A':>15}")
    print(f"{'Validation Accuracy':<30} {mlp_val_acc:>14.2f}% {gcn_val_acc:>14.2f}% {'N/A':>15}")
    print(f"{'Test Accuracy':<30} {mlp_test_acc:>14.2f}% {gcn_test_acc:>14.2f}% {gap:>+14.2f}%")
    print(f"{'Best Epoch':<30} {mlp_epoch:>15} {gcn_epoch:>15} {'N/A':>15}")
    print(f"{'Training Time (s)':<30} {mlp_time:>15.2f} {gcn_time:>15.2f} {'N/A':>15}")
    print("-" * 70)
    print()

    # Analysis
    print("="*70)
    print("                           ANALYSIS")
    print("="*70)
    print()

    if gap > 0:
        print(f"✓ GCN outperforms MLP by {gap:.2f}% on test accuracy!")
        print()
        print("This demonstrates the power of graph convolution:")
        print("  • GCN leverages neighbor information through adjacency matrix")
        print("  • Each node's feature is a weighted average of its neighbors")
        print("  • This 'homophily' assumption works well on citation networks:")
        print("    'Papers citing similar papers have similar topics'")
    else:
        print(f"✗ MLP matches or outperforms GCN by {-gap:.2f}%")
        print("  (This is unexpected - may indicate implementation issues)")

    print()

    # Training dynamics analysis
    train_gap = gcn_train_acc - mlp_train_acc
    test_gap = gcn_test_acc - mlp_test_acc

    print("Training vs Test Gap Analysis:")
    print(f"  • Train gap: {train_gap:+.2f}% (GCN {'better' if train_gap > 0 else 'worse'} on training)")
    print(f"  • Test gap:  {test_gap:+.2f}% (GCN {'better' if test_gap > 0 else 'worse'} on testing)")

    if abs(train_gap) < abs(test_gap):
        print("  • The test improvement is larger than train improvement,")
        print("    suggesting graph structure provides better generalization!")
    else:
        print("  • The train improvement is larger than test improvement,")
        print("    suggesting some overfitting to training nodes.")

    print()
    print("="*70)


def main():
    """Main execution function for ablation study."""
    args = parse_arguments()

    print("\n" + "="*70)
    print(f"              ABLATION STUDY: MLP vs GCN on {args.dataset.upper()}")
    print("="*70)
    print()
    print("Purpose: Demonstrate the importance of graph convolution by comparing")
    print("         a standard MLP (no graph structure) vs GCN (with neighbors).")
    print()
    print("Controlled Variables:")
    print("  • Same layer dimensions")
    print(f"  • Same hyperparameters: lr={args.lr}, weight_decay={args.weight_decay}, patience={args.patience}")
    print("  • Same data split proportions")
    print("  • Same random seed for reproducibility")
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

    print(f"Configuration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Device: {DEVICE}")
    print(f"  Hidden Units: {HIDDEN}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LR}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Patience: {PATIENCE}")
    print()

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

    print(f"Dataset loaded: {data['num_nodes']} nodes, {data['num_features']} features")
    print()

    # Create models
    gcn_model = GCN(
        nfeat=data['num_features'],
        nhid=HIDDEN,
        nclass=data['num_classes'],
        dropout=DROPOUT
    ).to(DEVICE)

    mlp_model = MLP(
        nfeat=data['num_features'],
        nhid=HIDDEN,
        nclass=data['num_classes'],
        dropout=DROPOUT
    ).to(DEVICE)

    print("Model Parameters:")
    gcn_params = sum(p.numel() for p in gcn_model.parameters())
    mlp_params = sum(p.numel() for p in mlp_model.parameters())
    print(f"  GCN: {gcn_params:,} parameters")
    print(f"  MLP: {mlp_params:,} parameters")
    print()

    # Train GCN
    gcn_results = train_model(
        model_name="GCN (Graph Convolutional Network)",
        model=gcn_model,
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
        seed=SEED
    )
    gcn_results['num_params'] = sum(p.numel() for p in gcn_model.parameters())

    # Train MLP (with same seed for fair comparison)
    mlp_results = train_model(
        model_name="MLP (Multi-Layer Perceptron)",
        model=mlp_model,
        x=x,
        adj=adj,  # Adj is ignored by MLP, but kept for API compatibility
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        patience=PATIENCE,
        seed=SEED
    )
    mlp_results['num_params'] = sum(p.numel() for p in mlp_model.parameters())

    # Print comparison table
    print_comparison_table(gcn_results, mlp_results, args.dataset,
                          data['num_features'], data['num_classes'], HIDDEN)

    return gcn_results, mlp_results


if __name__ == '__main__':
    main()
