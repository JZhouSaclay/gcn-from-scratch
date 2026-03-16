"""
Ablation Study: MLP vs GCN
==========================
Compare the performance of a standard MLP (no graph structure)
vs GCN (with graph convolution) on the Cora dataset.

This demonstrates the importance of neighbor aggregation in GCN.
Both models have identical architecture dimensions (1433->16->7)
and are trained with the same hyperparameters.

Usage:
    python run_ablation.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from src.data_loader import CoraDataLoader
from src.model import GCN, MLP
from src.train import train_gcn
from src.utils import set_seed


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


def print_comparison_table(gcn_results, mlp_results):
    """
    Print a beautiful comparison table of GCN vs MLP results.
    """
    print("\n" + "="*70)
    print("                    ABLATION STUDY: MLP vs GCN")
    print("="*70)
    print()
    print("Model Architecture Comparison:")
    print("-" * 70)
    print(f"{'Layer':<20} {'MLP':<25} {'GCN':<25}")
    print("-" * 70)
    print(f"{'Layer 1':<20} {'Linear(1433, 16)':<25} {'GraphConv(1433, 16)':<25}")
    print(f"{'Activation':<20} {'ReLU':<25} {'ReLU':<25}")
    print(f"{'Dropout':<20} {'p=0.5':<25} {'p=0.5':<25}")
    print(f"{'Layer 2':<20} {'Linear(16, 7)':<25} {'GraphConv(16, 7)':<25}")
    print(f"{'Parameters':<20} {'23,063':<25} {'23,063':<25}")
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
    print("\n" + "="*70)
    print("              ABLATION STUDY: MLP vs GCN on Cora")
    print("="*70)
    print()
    print("Purpose: Demonstrate the importance of graph convolution by comparing")
    print("         a standard MLP (no graph structure) vs GCN (with neighbors).")
    print()
    print("Controlled Variables:")
    print("  • Same architecture: 1433 -> 16 -> 7")
    print("  • Same hyperparameters: lr=0.01, weight_decay=5e-4, patience=50")
    print("  • Same data split: 140 train / 500 val / 1000 test")
    print("  • Same random seed for reproducibility")
    print()

    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    EPOCHS = 200
    LR = 0.01
    WEIGHT_DECAY = 5e-4
    PATIENCE = 50
    HIDDEN = 16
    DROPOUT = 0.5

    print(f"Configuration:")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LR}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  Patience: {PATIENCE}")
    print()

    # Load Cora dataset
    print("Loading Cora dataset...")
    loader = CoraDataLoader()
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

    # Print comparison table
    print_comparison_table(gcn_results, mlp_results)

    return gcn_results, mlp_results


if __name__ == '__main__':
    main()
