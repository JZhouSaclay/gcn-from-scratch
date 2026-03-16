"""
GCN Model Module
================
Full Graph Convolutional Network model implementation.

Architecture (as per Kipf & Welling paper):
    Input → GCN Layer 1 (ReLU + Dropout) → GCN Layer 2 → Output

Dimensions:
    - Input: [N, F] where F is input feature dimension
    - Hidden: [N, H] where H is hidden layer dimension (typically 16)
    - Output: [N, C] where C is number of classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle imports for both module usage and direct script execution
try:
    from .layers import GraphConvolution
except ImportError:
    from layers import GraphConvolution


class GCN(nn.Module):
    """
    2-layer Graph Convolutional Network for semi-supervised classification.

    This is the exact architecture used in the Kipf & Welling paper
    for citation network classification tasks.

    Architecture:
        Layer 1: X -> GraphConv -> ReLU -> Dropout -> H
        Layer 2: H -> GraphConv -> Output (logits)

    Note: LogSoftmax is NOT applied here; use CrossEntropyLoss in training loop.

    Attributes:
        gc1: First GraphConvolution layer (input → hidden)
        gc2: Second GraphConvolution layer (hidden → output)
        dropout: Dropout probability
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        """
        Initialize 2-layer GCN model.

        Args:
            nfeat: Number of input features per node
            nhid: Number of hidden units (typically 16 for Cora)
            nclass: Number of output classes
            dropout: Dropout probability (default: 0.5 as in paper)
        """
        super(GCN, self).__init__()

        # First GCN layer: nfeat -> nhid
        self.gc1 = GraphConvolution(nfeat, nhid, bias=True)

        # Second GCN layer: nhid -> nclass
        self.gc2 = GraphConvolution(nhid, nclass, bias=True)

        # Store dropout rate
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Forward pass through the full network.

        Args:
            x: Input feature matrix [N, nfeat]
            adj: Normalized adjacency matrix [N, N]

        Returns:
            Logits [N, nclass] (raw output, no softmax)
        """
        # Layer 1: GraphConv + ReLU + Dropout
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: GraphConv (output logits)
        x = self.gc2(x, adj)

        return x

    def predict(self, x, adj):
        """
        Make predictions without training-specific operations (no dropout).

        Args:
            x: Input feature matrix [N, nfeat]
            adj: Normalized adjacency matrix [N, N]

        Returns:
            Class predictions [N] (argmax of probabilities)
        """
        self.eval()  # Set to evaluation mode (disables dropout)
        with torch.no_grad():
            logits = self.forward(x, adj)
            predictions = logits.argmax(dim=1)
        return predictions


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for ablation study.

    This is a standard MLP with the same layer dimensions as GCN
    (nfeat -> nhid -> nclass) but WITHOUT using the adjacency matrix.
    It serves as a baseline to demonstrate the importance of graph
    structure information in GCN.

    Architecture:
        Layer 1: X -> Linear -> ReLU -> Dropout -> H
        Layer 2: H -> Linear -> Output (logits)

    Note: No graph convolution is performed. Each node is processed independently.
    """

    def __init__(self, nfeat, nhid, nclass, dropout=0.5):
        """
        Initialize 2-layer MLP model.

        Args:
            nfeat: Number of input features per node
            nhid: Number of hidden units (typically 16 for Cora)
            nclass: Number of output classes
            dropout: Dropout probability (default: 0.5)
        """
        super(MLP, self).__init__()

        # First linear layer: nfeat -> nhid
        self.fc1 = nn.Linear(nfeat, nhid, bias=True)

        # Second linear layer: nhid -> nclass
        self.fc2 = nn.Linear(nhid, nclass, bias=True)

        # Store dropout rate
        self.dropout = dropout

    def forward(self, x, adj=None):
        """
        Forward pass through the MLP.

        Args:
            x: Input feature matrix [N, nfeat]
            adj: Adjacency matrix (IGNORED - for API compatibility with GCN)

        Returns:
            Logits [N, nclass] (raw output, no softmax)
        """
        # Layer 1: Linear + ReLU + Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: Linear (output logits)
        x = self.fc2(x)

        return x

    def predict(self, x, adj=None):
        """
        Make predictions without training-specific operations (no dropout).

        Args:
            x: Input feature matrix [N, nfeat]
            adj: Adjacency matrix (IGNORED - for API compatibility with GCN)

        Returns:
            Class predictions [N] (argmax of probabilities)
        """
        self.eval()  # Set to evaluation mode (disables dropout)
        with torch.no_grad():
            logits = self.forward(x)
            predictions = logits.argmax(dim=1)
        return predictions


class DeepGCN(nn.Module):
    """
    Deep GCN with multiple hidden layers (for experimentation).

    Note: The original paper uses only 2 layers. Deeper GCNs may suffer
    from over-smoothing issues.
    """

    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5):
        """
        Args:
            nfeat: Number of input features
            nhid: Number of hidden units per layer
            nclass: Number of output classes
            nlayers: Total number of GCN layers (including output layer)
            dropout: Dropout probability
        """
        super(DeepGCN, self).__init__()

        assert nlayers >= 2, "DeepGCN requires at least 2 layers"

        self.layers = nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.layers.append(GraphConvolution(nfeat, nhid, bias=True))

        # Hidden layers
        for _ in range(nlayers - 2):
            self.layers.append(GraphConvolution(nhid, nhid, bias=True))

        # Output layer
        self.layers.append(GraphConvolution(nhid, nclass, bias=True))

    def forward(self, x, adj):
        """
        Forward pass for deep GCN.

        Args:
            x: Input feature matrix [N, nfeat]
            adj: Normalized adjacency matrix [N, N]

        Returns:
            Logits [N, nclass]
        """
        # All layers except the last one: GraphConv + ReLU + Dropout
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last layer: just GraphConv (output logits)
        x = self.layers[-1](x, adj)

        return x


if __name__ == '__main__':
    """Test GCN model with dummy data."""
    print("=" * 60)
    print("Testing GCN Model")
    print("=" * 60)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define dimensions
    n_nodes = 5
    n_features = 10
    n_hidden = 16
    n_classes = 2

    print(f"\nDummy data dimensions:")
    print(f"  Number of nodes: {n_nodes}")
    print(f"  Input features: {n_features}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  Output classes: {n_classes}")

    # Create dummy data
    # Random node features
    x = torch.randn(n_nodes, n_features)
    print(f"\nInput features shape: {x.shape}")

    # Create a simple adjacency matrix (normalized, with self-loops)
    # Simple chain graph: 0 -- 1 -- 2 -- 3 -- 4
    adj = torch.eye(n_nodes)
    adj[0, 1] = adj[1, 0] = 0.5
    adj[1, 2] = adj[2, 1] = 0.5
    adj[2, 3] = adj[3, 2] = 0.5
    adj[3, 4] = adj[4, 3] = 0.5
    print(f"Adjacency matrix shape: {adj.shape}")

    # Test 1: Create GCN model
    print("\n" + "-" * 60)
    print("[Test 1] Create GCN model and check forward pass")
    print("-" * 60)

    model = GCN(nfeat=n_features, nhid=n_hidden, nclass=n_classes, dropout=0.5)
    print(f"Model created:")
    print(f"  gc1: {model.gc1}")
    print(f"  gc2: {model.gc2}")
    print(f"  dropout: {model.dropout}")

    # Test 2: Forward pass in training mode
    print("\n[Test 2] Forward pass (training mode)")
    model.train()
    output_train = model(x, adj)
    print(f"Output shape: {output_train.shape}")
    print(f"Expected shape: torch.Size([{n_nodes}, {n_classes}])")
    assert output_train.shape == torch.Size([n_nodes, n_classes]), "Output shape mismatch!"
    print(f"Output values (first 3 nodes):\n{output_train[:3]}")
    print("✓ Training forward pass passed")

    # Test 3: Forward pass in eval mode
    print("\n[Test 3] Forward pass (eval mode)")
    model.eval()
    output_eval = model(x, adj)
    print(f"Output shape: {output_eval.shape}")
    # Output should be different from training due to no dropout
    assert not torch.equal(output_train, output_eval), "Dropout not working (outputs identical)!"
    print("✓ Eval mode produces different output (dropout disabled)")

    # Test 4: Prediction method
    print("\n[Test 4] Predict method")
    predictions = model.predict(x, adj)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions.tolist()}")
    assert predictions.shape == torch.Size([n_nodes]), "Prediction shape mismatch!"
    assert predictions.dtype == torch.long, "Predictions should be long dtype!"
    print("✓ Prediction method passed")

    # Test 5: Gradient flow
    print("\n[Test 5] Gradient flow")
    model.train()
    output = model(x, adj)
    loss = output.sum()
    loss.backward()

    print("Gradients computed:")
    print(f"  gc1.weight.grad shape: {model.gc1.weight.grad.shape}")
    print(f"  gc1.bias.grad shape: {model.gc1.bias.grad.shape}")
    print(f"  gc2.weight.grad shape: {model.gc2.weight.grad.shape}")
    print(f"  gc2.bias.grad shape: {model.gc2.bias.grad.shape}")

    assert model.gc1.weight.grad is not None, "gc1 weight gradient missing!"
    assert model.gc2.weight.grad is not None, "gc2 weight gradient missing!"
    print("✓ Gradient flow test passed")

    # Test 6: Test with different dropout values
    print("\n[Test 6] Different dropout values")
    model_no_dropout = GCN(nfeat=n_features, nhid=n_hidden, nclass=n_classes, dropout=0.0)
    model_no_dropout.train()
    output_no_dropout = model_no_dropout(x, adj)
    print(f"Output with dropout=0.0: shape={output_no_dropout.shape}")
    print("✓ Dropout=0 test passed")

    print("\n" + "=" * 60)
    print("All GCN model tests passed!")
    print("=" * 60)

    # Summary
    print("\n[Summary]")
    print(f"  Input feature shape:  {x.shape}")
    print(f"  Adjacency shape:      {adj.shape}")
    print(f"  Hidden layer output:  torch.Size([{n_nodes}, {n_hidden}]) (intermediate)")
    print(f"  Final output shape:   {output_train.shape}")
    print(f"  Predictions shape:    {predictions.shape}")
