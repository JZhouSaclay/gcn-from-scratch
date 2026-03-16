"""
Utility Functions Module
========================
Helper functions for graph operations and matrix computations.

All functions use only basic PyTorch tensor operations.
"""

import torch
import torch.nn.functional as F
import numpy as np


def add_self_loops(adj):
    """
    Add self-loops to the adjacency matrix.

    Operation: A_tilde = A + I

    Args:
        adj: Adjacency matrix [N, N]

    Returns:
        Adjacency matrix with self-loops [N, N]
    """
    num_nodes = adj.shape[0]
    # A_tilde = A + I
    adj_tilde = adj + torch.eye(num_nodes, dtype=adj.dtype, device=adj.device)
    return adj_tilde


def symmetric_normalize(adj):
    """
    Compute symmetrically normalized adjacency matrix.

    Formula from Kipf & Welling paper:
        A_hat = D_tilde^(-1/2) @ A_tilde @ D_tilde^(-1/2)

    where:
        - A_tilde = A + I (adjacency with self-loops)
        - D_tilde is the diagonal degree matrix of A_tilde
        - D_tilde^(-1/2) = diag(1/sqrt(degree))

    Args:
        adj: Adjacency matrix [N, N] (without self-loops)

    Returns:
        Normalized adjacency matrix [N, N]
    """
    # Step 1: Add self-loops to get A_tilde
    adj_tilde = add_self_loops(adj)

    # Step 2: Compute degree of each node: D_ii = sum_j(A_tilde[i,j])
    # Result: degree[i] = sum of row i in adj_tilde
    degree = torch.sum(adj_tilde, dim=1)

    # Step 3: Compute D^(-1/2): take inverse square root of degrees
    # Add small epsilon to avoid division by zero
    eps = 1e-8
    d_inv_sqrt = torch.pow(degree + eps, -0.5)

    # Step 4: Apply normalization using broadcasting
    # A_hat = D^(-1/2) @ A_tilde @ D^(-1/2)
    # This can be done as: (D^(-1/2)[:, None] * A_tilde) * D^(-1/2)[None, :]
    # Or equivalently: d_inv_sqrt.view(-1, 1) * adj_tilde * d_inv_sqrt.view(1, -1)
    adj_norm = d_inv_sqrt.view(-1, 1) * adj_tilde * d_inv_sqrt.view(1, -1)

    return adj_norm


def normalize_adjacency(adj):
    """
    Convenience function: add self-loops and symmetrically normalize.

    This is the complete normalization used in GCN paper.

    Args:
        adj: Raw adjacency matrix [N, N]

    Returns:
        Normalized adjacency matrix [N, N]
    """
    return symmetric_normalize(adj)


def sparse_to_dense(edge_index, num_nodes):
    """
    Convert edge list to dense adjacency matrix.

    Args:
        edge_index: Edge list [2, E] where each column is (source, target)
        num_nodes: Number of nodes

    Returns:
        Dense adjacency matrix [N, N]
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # edge_index shape: [2, E]
    # edge_index[0]: source nodes
    # edge_index[1]: target nodes
    if isinstance(edge_index, torch.Tensor):
        src = edge_index[0].long()
        dst = edge_index[1].long()
    else:
        src = torch.tensor([e[0] for e in edge_index], dtype=torch.long)
        dst = torch.tensor([e[1] for e in edge_index], dtype=torch.long)

    # Set edges (undirected graph: add both directions)
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0

    return adj


def accuracy(pred, labels, mask=None):
    """
    Calculate classification accuracy.

    Args:
        pred: Predictions [N, C] or [N] (if already argmaxed)
        labels: Ground truth labels [N]
        mask: Optional boolean mask [N] to select subset of nodes

    Returns:
        Accuracy as a float
    """
    # If pred is [N, C], take argmax to get predicted class
    if pred.dim() > 1:
        pred = pred.argmax(dim=1)

    correct = (pred == labels).float()

    if mask is not None:
        correct = correct[mask]
        return correct.sum().item() / mask.sum().item()
    else:
        return correct.mean().item()


def masked_softmax_cross_entropy(logits, labels, mask):
    """
    Compute masked softmax cross-entropy loss.

    Only compute loss on nodes specified by the mask.

    Args:
        logits: Model output before softmax [N, C]
        labels: Ground truth labels [N]
        mask: Boolean mask [N] indicating which nodes to include

    Returns:
        Loss value (scalar tensor)
    """
    # Apply log_softmax for numerical stability
    log_probs = F.log_softmax(logits, dim=1)

    # Select only masked nodes
    masked_log_probs = log_probs[mask]
    masked_labels = labels[mask]

    # Compute negative log likelihood loss
    loss = F.nll_loss(masked_log_probs, masked_labels)

    return loss


def normalize_features(x):
    """
    Row-normalize feature matrix (L2 normalization).

    Args:
        x: Feature matrix [N, F]

    Returns:
        Normalized feature matrix [N, F]
    """
    # Compute L2 norm for each row
    row_norms = torch.norm(x, p=2, dim=1, keepdim=True)
    # Add epsilon to avoid division by zero
    eps = 1e-8
    x_norm = x / (row_norms + eps)
    return x_norm


def set_seed(seed=42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    """Test utility functions."""
    print("=" * 60)
    print("Testing utils.py functions")
    print("=" * 60)

    # Test 1: add_self_loops
    print("\n[Test 1] add_self_loops")
    adj = torch.zeros((3, 3))
    adj[0, 1] = 1.0
    adj[1, 0] = 1.0  # undirected edge 0-1
    adj[1, 2] = 1.0
    adj[2, 1] = 1.0  # undirected edge 1-2
    print(f"Original adjacency:\n{adj}")
    adj_tilde = add_self_loops(adj)
    print(f"With self-loops:\n{adj_tilde}")
    assert torch.all(torch.diag(adj_tilde) == 1.0), "Self-loops not added correctly!"
    print("✓ add_self_loops passed")

    # Test 2: symmetric_normalize
    print("\n[Test 2] symmetric_normalize")
    adj_norm = symmetric_normalize(adj)
    print(f"Normalized adjacency shape: {adj_norm.shape}")
    print(f"Normalized adjacency:\n{adj_norm}")
    # Check symmetry
    assert torch.allclose(adj_norm, adj_norm.T), "Normalized matrix not symmetric!"
    # Check row sums (should be close to 1 for normalized random walk)
    row_sums = torch.sum(adj_norm, dim=1)
    print(f"Row sums: {row_sums}")
    print("✓ symmetric_normalize passed")

    # Test 3: sparse_to_dense
    print("\n[Test 3] sparse_to_dense")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # 3 edges
    dense_adj = sparse_to_dense(edge_index, num_nodes=3)
    print(f"Edge list:\n{edge_index}")
    print(f"Dense adjacency:\n{dense_adj}")
    assert dense_adj.shape == (3, 3), "Wrong shape!"
    print("✓ sparse_to_dense passed")

    # Test 4: normalize_features
    print("\n[Test 4] normalize_features")
    features = torch.tensor([[3.0, 4.0], [1.0, 1.0], [0.0, 5.0]])
    print(f"Original features:\n{features}")
    features_norm = normalize_features(features)
    print(f"Normalized features:\n{features_norm}")
    # Check L2 norm is 1 for each row
    row_norms = torch.norm(features_norm, p=2, dim=1)
    print(f"Row norms after normalization: {row_norms}")
    assert torch.allclose(row_norms, torch.ones_like(row_norms)), "Features not normalized!"
    print("✓ normalize_features passed")

    # Test 5: accuracy
    print("\n[Test 5] accuracy")
    logits = torch.tensor([[1.0, 2.0], [3.0, 1.0], [1.0, 4.0]])  # predicts: [1, 0, 1]
    labels = torch.tensor([1, 0, 1])  # actual: [1, 0, 1] - all correct!
    mask = torch.tensor([True, True, False])  # only check first 2
    acc_all = accuracy(logits, labels)
    acc_masked = accuracy(logits, labels, mask)
    print(f"Predictions: {logits.argmax(dim=1).tolist()}")
    print(f"Labels: {labels.tolist()}")
    print(f"Accuracy (all): {acc_all:.2f}")
    print(f"Accuracy (masked): {acc_masked:.2f}")
    assert acc_all == 1.0, f"Expected 1.0, got {acc_all}"
    assert acc_masked == 1.0, f"Expected 1.0, got {acc_masked}"
    print("✓ accuracy passed")

    print("\n" + "=" * 60)
    print("All utils.py tests passed!")
    print("=" * 60)
