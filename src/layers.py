"""
GCN Layer Module
================
Implementation of the Graph Convolutional Network layer from scratch.

Paper: "Semi-Supervised Classification with Graph Convolutional Networks"
       by Kipf & Welling (ICLR 2017)

Key formula:
    H^(l+1) = σ(A_hat @ H^(l) @ W^(l))

where:
    - A_hat is the normalized adjacency matrix (pre-computed)
    - H^(l) is the node feature matrix at layer l
    - W^(l) is the learnable weight matrix
    - σ is the activation function (e.g., ReLU)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    """
    Graph Convolutional Layer (GCN Layer).

    This layer performs:
        H = A_hat @ (X @ W) + bias

    where:
        - X: input features [N, in_features]
        - A_hat: normalized adjacency matrix [N, N]
        - W: learnable weight matrix [in_features, out_features]
        - bias: learnable bias [out_features]

    Attributes:
        weight: Learnable weight matrix [in_features, out_features]
        bias: Learnable bias [out_features]
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize Graph Convolution layer.

        Args:
            in_features: Size of input features per node
            out_features: Size of output features per node
            bias: Whether to include learnable bias (default: True)
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrix as a learnable parameter
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Initialize bias if required
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights using Kaiming (He) uniform initialization.

        This is similar to PyTorch's default initialization for Linear layers.
        For GCN, this helps with training stability.
        """
        # Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # Initialize bias using the same distribution as PyTorch Linear
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, adj):
        """
        Forward pass of Graph Convolution layer.

        Formula: H = adj @ (x @ W) + bias

        Args:
            x: Input feature matrix [N, in_features]
            adj: Normalized adjacency matrix [N, N] (pre-computed A_hat)

        Returns:
            Output feature matrix [N, out_features]
        """
        # Step 1: Feature transformation: x @ W
        # [N, in_features] @ [in_features, out_features] -> [N, out_features]
        support = torch.matmul(x, self.weight)

        # Step 2: Neighborhood aggregation: adj @ support
        # [N, N] @ [N, out_features] -> [N, out_features]
        output = torch.matmul(adj, support)

        # Step 3: Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None})")


class GCNLayer(nn.Module):
    """
    Higher-level GCN layer with built-in activation and dropout.

    This is a convenience wrapper that combines:
        - Graph Convolution
        - Activation function (optional)
        - Dropout (optional, during training only)
    """

    def __init__(self, in_features, out_features, bias=True, activation=None, dropout=0.0):
        """
        Initialize GCN layer with activation and dropout.

        Args:
            in_features: Size of input features per node
            out_features: Size of output features per node
            bias: Whether to include learnable bias
            activation: Activation function ('relu', 'softmax', or None)
            dropout: Dropout probability (applied during training)
        """
        super(GCNLayer, self).__init__()
        self.gc = GraphConvolution(in_features, out_features, bias)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, adj_norm):
        """
        Forward pass with activation and dropout.

        Args:
            x: Input feature matrix [N, in_features]
            adj_norm: Normalized adjacency matrix [N, N]

        Returns:
            Output feature matrix [N, out_features]
        """
        # Graph convolution
        x = self.gc(x, adj_norm)

        # Activation
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'softmax':
            x = F.softmax(x, dim=1)

        # Dropout (only during training)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x


if __name__ == '__main__':
    """Test GraphConvolution layer."""
    print("=" * 60)
    print("Testing GraphConvolution Layer")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Test 1: Basic forward pass
    print("\n[Test 1] Basic forward pass")
    n_nodes = 5
    in_features = 10
    out_features = 16

    # Create dummy data
    x = torch.randn(n_nodes, in_features)
    # Create a simple adjacency matrix with self-loops
    adj = torch.eye(n_nodes)
    adj[0, 1] = adj[1, 0] = 1.0
    adj[1, 2] = adj[2, 1] = 1.0
    adj[2, 3] = adj[3, 2] = 1.0

    print(f"Input x shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")

    # Create layer
    gc = GraphConvolution(in_features, out_features, bias=True)
    print(f"\nLayer: {gc}")

    # Forward pass
    out = gc(x, adj)
    print(f"Output shape: {out.shape}")
    print(f"Expected shape: torch.Size([{n_nodes}, {out_features}])")
    assert out.shape == torch.Size([n_nodes, out_features]), "Output shape mismatch!"
    print("✓ Shape test passed")

    # Test 2: Verify computation graph
    print("\n[Test 2] Gradient flow check")
    loss = out.sum()
    loss.backward()
    print(f"Weight grad shape: {gc.weight.grad.shape}")
    print(f"Bias grad shape: {gc.bias.grad.shape}")
    assert gc.weight.grad is not None, "Weight gradient not computed!"
    assert gc.bias.grad is not None, "Bias gradient not computed!"
    print("✓ Gradient flow test passed")

    # Test 3: No bias
    print("\n[Test 3] Without bias")
    gc_no_bias = GraphConvolution(in_features, out_features, bias=False)
    out_no_bias = gc_no_bias(x, adj)
    assert gc_no_bias.bias is None
    assert out_no_bias.shape == torch.Size([n_nodes, out_features])
    print("✓ No bias test passed")

    # Test 4: Reset parameters
    print("\n[Test 4] Reset parameters")
    old_weight = gc.weight.data.clone()
    gc.reset_parameters()
    new_weight = gc.weight.data
    assert not torch.equal(old_weight, new_weight), "Weights not reset!"
    print("✓ Reset parameters test passed")

    # Test 5: Verify computation formula
    print("\n[Test 5] Verify H = adj @ (x @ W) + bias")
    gc_test = GraphConvolution(in_features, out_features, bias=True)
    # Manually compute expected output
    support = torch.matmul(x, gc_test.weight)
    expected = torch.matmul(adj, support)
    if gc_test.bias is not None:
        expected = expected + gc_test.bias
    actual = gc_test(x, adj)
    assert torch.allclose(expected, actual, atol=1e-6), "Formula implementation incorrect!"
    print("✓ Formula verification passed")

    print("\n" + "=" * 60)
    print("All layer tests passed!")
    print("=" * 60)
