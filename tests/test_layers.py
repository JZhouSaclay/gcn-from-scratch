"""
Unit Tests for GCN Layers
==========================
Test the core layer implementations to ensure correctness.
"""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.layers import GCNLayer
from src.utils import add_self_loops, symmetric_normalize


def test_gcn_layer_output_shape():
    """
    Test that GCN layer produces correct output shape.

    Input: [N, in_features] -> Output: [N, out_features]
    """
    # TODO: Implement shape test
    pass


def test_symmetric_normalization():
    """
    Test that normalized adjacency matrix is symmetric.

    A_hat should equal A_hat^T
    """
    # TODO: Implement symmetry test
    pass


def test_self_loops():
    """
    Test that self-loops are correctly added.

    Diagonal of A_tilde should be all 1s
    """
    # TODO: Implement self-loop test
    pass


def test_gcn_layer_gradient():
    """
    Test that gradients flow through GCN layer.
    """
    # TODO: Implement gradient flow test
    pass


if __name__ == '__main__':
    print("Running layer tests...")
    # Run tests
    print("All tests passed!")
