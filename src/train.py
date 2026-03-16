"""
Training and Evaluation Module
==============================
Implements the training loop, evaluation metrics, and early stopping.

Training strategy (as per paper):
    - Adam optimizer with learning rate 0.01
    - Weight decay (L2 regularization) 5e-4
    - Early stopping based on validation loss (patience=10)
    - Training on 20 samples per class only (semi-supervised)
"""

import time
import torch
import torch.nn.functional as F
import torch.optim as optim

# Handle imports for both module usage and direct script execution
try:
    from .utils import accuracy
except ImportError:
    from utils import accuracy


def train_gcn(model, x, adj, labels, train_mask, val_mask, test_mask,
              epochs=200, lr=0.01, weight_decay=5e-4, patience=10, verbose=True):
    """
    Train GCN model with early stopping.

    This function implements the complete training loop with:
    - Adam optimizer
    - Cross-entropy loss with masking (semi-supervised)
    - Validation-based early stopping
    - Periodic logging

    Args:
        model: GCN model instance
        x: Node features [N, F]
        adj: Normalized adjacency matrix [N, N]
        labels: Ground truth labels [N]
        train_mask: Training mask [N]
        val_mask: Validation mask [N]
        test_mask: Test mask [N]
        epochs: Number of training epochs (default: 200)
        lr: Learning rate (default: 0.01)
        weight_decay: L2 regularization (default: 5e-4)
        patience: Early stopping patience (default: 10)
        verbose: Whether to print progress (default: True)

    Returns:
        results: Dictionary with final metrics and history
    """
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Move data to same device as model
    device = next(model.parameters()).device
    x = x.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    # Early stopping variables
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None

    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    if verbose:
        print("\n" + "=" * 60)
        print("Training Started")
        print("=" * 60)
        print(f"Epochs: {epochs} | LR: {lr} | Weight Decay: {weight_decay}")
        print(f"Patience: {patience} | Device: {device}")
        print("-" * 60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # ==================== Training ====================
        model.train()
        optimizer.zero_grad()

        # Forward pass
        logits = model(x, adj)

        # Compute loss only on training nodes (semi-supervised!)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        with torch.no_grad():
            train_acc = accuracy(logits[train_mask], labels[train_mask])

        # ==================== Validation ====================
        model.eval()
        with torch.no_grad():
            logits = model(x, adj)

            # Validation loss and accuracy (only on validation nodes)
            val_loss = F.cross_entropy(logits[val_mask], labels[val_mask]).item()
            val_acc = accuracy(logits[val_mask], labels[val_mask])

        # Store history
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # ==================== Logging ====================
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        # ==================== Early Stopping ====================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            # Save best model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

    training_time = time.time() - start_time

    # ==================== Final Evaluation ====================
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        logits = model(x, adj)

        # Test accuracy
        test_acc = accuracy(logits[test_mask], labels[test_mask])

        # Also compute train and val accuracy with best model
        final_train_acc = accuracy(logits[train_mask], labels[train_mask])
        final_val_acc = accuracy(logits[val_mask], labels[val_mask])

    if verbose:
        print("\n" + "=" * 60)
        print("Training Completed")
        print("=" * 60)
        print(f"Best Epoch: {best_epoch}")
        print(f"Training Time: {training_time:.2f}s")
        print(f"Final Train Acc: {final_train_acc:.4f}")
        print(f"Final Val Acc:   {final_val_acc:.4f}")
        print(f"Final Test Acc:  {test_acc:.4f}")
        print("=" * 60)

    results = {
        'model': model,
        'test_acc': test_acc,
        'train_acc': final_train_acc,
        'val_acc': final_val_acc,
        'best_epoch': best_epoch,
        'training_time': training_time,
        'history': history
    }

    return results


if __name__ == '__main__':
    """Test train_gcn with dummy data."""
    print("=" * 60)
    print("Testing train_gcn function")
    print("=" * 60)

    try:
        from model import GCN
        from utils import set_seed
    except ImportError:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from model import GCN
        from utils import set_seed

    # Set seed for reproducibility
    set_seed(42)

    # Create dummy data
    n_nodes = 100
    n_features = 10
    n_hidden = 16
    n_classes = 3

    # Dummy features and adjacency
    x = torch.randn(n_nodes, n_features)
    adj = torch.eye(n_nodes)
    # Add some random edges
    for _ in range(50):
        i, j = torch.randint(0, n_nodes, (2,))
        adj[i, j] = adj[j, i] = 1.0

    # Dummy labels
    labels = torch.randint(0, n_classes, (n_nodes,))

    # Dummy masks (30 train, 20 val, 30 test)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[:30] = True
    val_mask[30:50] = True
    test_mask[50:80] = True

    print(f"\nDummy Dataset:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Features: {n_features}")
    print(f"  Classes: {n_classes}")
    print(f"  Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | Test: {test_mask.sum().item()}")

    # Create model
    model = GCN(n_features, n_hidden, n_classes, dropout=0.5)

    # Train
    print("\nStarting training...")
    results = train_gcn(
        model, x, adj, labels, train_mask, val_mask, test_mask,
        epochs=50, lr=0.01, weight_decay=5e-4, patience=10, verbose=True
    )

    print("\n✓ train_gcn test completed successfully!")
