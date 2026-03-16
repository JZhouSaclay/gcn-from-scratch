"""
GCN From Scratch - Main Entry Point
====================================
Run this script to train and evaluate the GCN model on citation networks.

Usage:
    python main.py --dataset cora --epochs 200 --lr 0.01
    python main.py --dataset citeseer --hidden 16 --dropout 0.5

Arguments:
    --dataset: Dataset to use (cora, citeseer, pubmed)
    --hidden: Number of hidden units (default: 16)
    --epochs: Number of training epochs (default: 200)
    --lr: Learning rate (default: 0.01)
    --weight_decay: L2 regularization (default: 5e-4)
    --dropout: Dropout rate (default: 0.5)
    --seed: Random seed (default: 42)
    --patience: Early stopping patience (default: 10)
"""

import argparse
import torch

from src.data_loader import CoraDataLoader, CiteSeerDataLoader
from src.model import GCN
from src.train import train_gcn
from src.utils import set_seed


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='GCN from Scratch - Semi-supervised Node Classification'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='cora',
        choices=['cora', 'citeseer', 'pubmed'],
        help='Dataset to use (default: cora)'
    )

    parser.add_argument(
        '--hidden',
        type=int,
        default=16,
        help='Number of hidden units (default: 16)'
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
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )

    return parser.parse_args()


def load_data(dataset_name):
    """Load specified dataset."""
    if dataset_name == 'cora':
        loader = CoraDataLoader()
    elif dataset_name == 'citeseer':
        loader = CiteSeerDataLoader()
    else:
        raise ValueError(f"Dataset {dataset_name} not implemented yet")

    return loader.load()


def main():
    """Main execution function."""
    args = parse_arguments()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("GCN From Scratch - Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Hidden Units: {args.hidden}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[Step 1/4] Loading dataset...")
    data = load_data(args.dataset)

    print(f"Dataset Loaded:")
    print(f"  Nodes: {data['num_nodes']}")
    print(f"  Features: {data['num_features']}")
    print(f"  Classes: {data['num_classes']}")
    print(f"  Train: {data['train_mask'].sum().item()} nodes")
    print(f"  Val: {data['val_mask'].sum().item()} nodes")
    print(f"  Test: {data['test_mask'].sum().item()} nodes")

    # Move data to device
    x = data['x'].to(device)
    adj = data['adj'].to(device)
    labels = data['y'].to(device)
    train_mask = data['train_mask'].to(device)
    val_mask = data['val_mask'].to(device)
    test_mask = data['test_mask'].to(device)

    # Step 2: Build model
    print("\n[Step 2/4] Building GCN model...")
    model = GCN(
        nfeat=data['num_features'],
        nhid=args.hidden,
        nclass=data['num_classes'],
        dropout=args.dropout
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture:")
    print(f"  Input Layer: {data['num_features']} -> {args.hidden}")
    print(f"  Hidden Layer: {args.hidden} -> {data['num_classes']}")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # Step 3: Train model
    print("\n[Step 3/4] Starting training...")
    results = train_gcn(
        model=model,
        x=x,
        adj=adj,
        labels=labels,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        verbose=True
    )

    # Step 4: Final evaluation
    print("\n[Step 4/4] Final Evaluation on Test Set")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        logits = model(x, adj)

        # Get predictions
        predictions = logits.argmax(dim=1)

        # Compute test accuracy
        test_correct = (predictions[test_mask] == labels[test_mask]).sum().item()
        test_total = test_mask.sum().item()
        test_acc = test_correct / test_total

        print(f"\nTest Set Results:")
        print(f"  Correct Predictions: {test_correct}/{test_total}")
        print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

        # Compare with paper
        paper_acc = 0.815  # Kipf & Welling paper result
        diff = (test_acc - paper_acc) * 100
        print(f"\nComparison with Paper:")
        print(f"  Paper (Kipf & Welling 2017): {paper_acc*100:.1f}%")
        print(f"  Our Implementation:          {test_acc*100:.2f}%")
        print(f"  Difference:                  {diff:+.2f}%")

    print("\n" + "=" * 60)
    print("Training Pipeline Completed!")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()
