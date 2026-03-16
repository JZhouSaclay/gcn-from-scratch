"""
Data Loader Module
==================
Handles loading and preprocessing of Cora/CiteSeer/PubMed datasets.

All operations are implemented using basic PyTorch tensors without
PyTorch Geometric or DGL dependencies.
"""

import os
import sys
import urllib.request
import tarfile
import numpy as np
import torch

# Handle imports for both module usage and direct script execution
try:
    from .utils import normalize_features, normalize_adjacency, sparse_to_dense
except ImportError:
    from utils import normalize_features, normalize_adjacency, sparse_to_dense


class CoraDataLoader:
    """
    Data loader for the Cora citation network dataset.

    The Cora dataset consists of:
    - x: Node features (bag-of-words representation of documents)
    - y: Node labels (paper categories)
    - edge_index: Citation links between papers

    Dataset source: https://linqs.soe.ucsc.edu/data
    """

    URL = 'https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz'

    def __init__(self, root=None):
        """
        Args:
            root: Root directory where the dataset will be stored.
                  If None, defaults to 'gcn-from-scratch/data/cora' relative to this file.
        """
        if root is None:
            # Get the directory of this file (src/)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root (gcn-from-scratch/)
            project_root = os.path.dirname(current_file_dir)
            # Set data directory to project_root/data/cora
            root = os.path.join(project_root, 'data', 'cora')

        self.root = root
        self.raw_dir = os.path.join(root, 'raw')

    def download(self):
        """Download the Cora dataset if not already present."""
        if os.path.exists(self.raw_dir):
            print(f"Dataset already exists at {self.raw_dir}")
            return

        print(f"Downloading Cora dataset from {self.URL}...")
        os.makedirs(self.root, exist_ok=True)

        # Download the tar.gz file
        tar_path = os.path.join(self.root, 'cora.tgz')
        urllib.request.urlretrieve(self.URL, tar_path)
        print(f"Downloaded to {tar_path}")

        # Extract the archive
        print("Extracting archive...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.root)
        print(f"Extracted to {self.root}")

        # Move files from cora/ subdirectory to raw/
        cora_subdir = os.path.join(self.root, 'cora')
        if os.path.exists(cora_subdir):
            os.rename(cora_subdir, self.raw_dir)
            print(f"Moved to {self.raw_dir}")

        # Remove the tar file
        os.remove(tar_path)
        print("Download complete!")

    def parse_content(self, filepath):
        """
        Parse the .content file to extract features and labels.

        Format: <paper_id> <feature_values> <class_label>

        Args:
            filepath: Path to the .content file

        Returns:
            features: Tensor of shape [N, F]
            labels: Tensor of shape [N]
            paper_ids: List of paper IDs for indexing
            paper_id_to_idx: Dictionary mapping paper ID to node index
            class_names: List of unique class names
        """
        print(f"Parsing content file: {filepath}")

        paper_ids = []
        features_list = []
        labels_list = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                paper_id = parts[0]
                feature_values = [int(x) for x in parts[1:-1]]
                class_label = parts[-1]

                paper_ids.append(paper_id)
                features_list.append(feature_values)
                labels_list.append(class_label)

        # Convert to numpy arrays
        features = np.array(features_list, dtype=np.float32)

        # Create label mapping
        unique_classes = sorted(list(set(labels_list)))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = np.array([class_to_idx[label] for label in labels_list], dtype=np.int64)

        # Create paper ID to index mapping
        paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}

        # Convert to PyTorch tensors
        features = torch.from_numpy(features)
        labels = torch.from_numpy(labels)

        print(f"  Loaded {len(paper_ids)} papers")
        print(f"  Feature dimension: {features.shape[1]}")
        print(f"  Number of classes: {len(unique_classes)}")

        return features, labels, paper_ids, paper_id_to_idx, unique_classes

    def parse_cites(self, filepath, paper_id_to_idx):
        """
        Parse the .cites file to build edge list.

        Format: <cited_paper_id> <citing_paper_id>
        (Note: direction is cited -> citing, meaning paper 2 cites paper 1)

        Args:
            filepath: Path to the .cites file
            paper_id_to_idx: Mapping from paper ID to node index

        Returns:
            edge_index: Tensor of shape [2, E]
        """
        print(f"Parsing cites file: {filepath}")

        edges = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cited_id = parts[0]
                    citing_id = parts[1]

                    # Only add edge if both papers are in our mapping
                    if cited_id in paper_id_to_idx and citing_id in paper_id_to_idx:
                        src = paper_id_to_idx[cited_id]
                        dst = paper_id_to_idx[citing_id]
                        edges.append((src, dst))

        # Convert to tensor [2, E]
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        print(f"  Loaded {len(edges)} citation edges")

        return edge_index

    def load(self):
        """
        Load and preprocess the complete dataset.

        Returns:
            data_dict: Dictionary containing:
                - 'x': Node features [N, F]
                - 'y': Node labels [N]
                - 'adj': Normalized adjacency matrix [N, N]
                - 'train_mask': Boolean mask for training nodes [N]
                - 'val_mask': Boolean mask for validation nodes [N]
                - 'test_mask': Boolean mask for test nodes [N]
                - 'num_features': Number of features (F)
                - 'num_classes': Number of classes (C)
        """
        # Download dataset if not present
        self.download()

        # File paths
        content_file = os.path.join(self.raw_dir, 'cora.content')
        cites_file = os.path.join(self.raw_dir, 'cora.cites')

        # Parse files
        features, labels, paper_ids, paper_id_to_idx, class_names = self.parse_content(content_file)
        edge_index = self.parse_cites(cites_file, paper_id_to_idx)

        num_nodes = features.shape[0]

        # Build dense adjacency matrix
        adj = sparse_to_dense(edge_index, num_nodes)

        # Normalize features (row-wise L2 normalization)
        features = normalize_features(features)

        # Normalize adjacency matrix (symmetric normalization with self-loops)
        adj_norm = normalize_adjacency(adj)

        # Create train/val/test masks
        train_mask, val_mask, test_mask = create_masks(num_nodes, labels, num_classes=len(class_names))

        data_dict = {
            'x': features,
            'y': labels,
            'adj': adj_norm,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'num_features': features.shape[1],
            'num_classes': len(class_names),
            'num_nodes': num_nodes,
            'class_names': class_names
        }

        return data_dict


class CiteSeerDataLoader:
    """Data loader for CiteSeer dataset (similar structure to Cora)."""

    URL = 'https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz'

    def __init__(self, root=None):
        if root is None:
            # Get the directory of this file (src/)
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root (gcn-from-scratch/)
            project_root = os.path.dirname(current_file_dir)
            # Set data directory to project_root/data/citeseer
            root = os.path.join(project_root, 'data', 'citeseer')

        self.root = root
        self.raw_dir = os.path.join(root, 'raw')

    def download(self):
        """Download CiteSeer dataset."""
        if os.path.exists(self.raw_dir):
            return

        print(f"Downloading CiteSeer dataset...")
        os.makedirs(self.root, exist_ok=True)

        tar_path = os.path.join(self.root, 'citeseer.tgz')
        urllib.request.urlretrieve(self.URL, tar_path)

        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.root)

        citeseer_subdir = os.path.join(self.root, 'citeseer')
        if os.path.exists(citeseer_subdir):
            os.rename(citeseer_subdir, self.raw_dir)

        os.remove(tar_path)
        print("Download complete!")

    def load(self):
        """Load CiteSeer dataset."""
        self.download()

        content_file = os.path.join(self.raw_dir, 'citeseer.content')
        cites_file = os.path.join(self.raw_dir, 'citeseer.cites')

        # Parse using same logic as Cora
        paper_ids = []
        features_list = []
        labels_list = []

        with open(content_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3:
                    continue
                paper_id = parts[0]
                feature_values = [int(x) for x in parts[1:-1]]
                class_label = parts[-1]

                paper_ids.append(paper_id)
                features_list.append(feature_values)
                labels_list.append(class_label)

        features = torch.tensor(features_list, dtype=torch.float32)

        unique_classes = sorted(list(set(labels_list)))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        labels = torch.tensor([class_to_idx[label] for label in labels_list], dtype=torch.long)

        paper_id_to_idx = {pid: idx for idx, pid in enumerate(paper_ids)}

        # Parse cites
        edges = []
        with open(cites_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cited_id = parts[0]
                    citing_id = parts[1]
                    if cited_id in paper_id_to_idx and citing_id in paper_id_to_idx:
                        src = paper_id_to_idx[cited_id]
                        dst = paper_id_to_idx[citing_id]
                        edges.append((src, dst))

        num_nodes = len(paper_ids)
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.zeros((2, 0), dtype=torch.long)

        adj = sparse_to_dense(edge_index, num_nodes)
        features = normalize_features(features)
        adj_norm = normalize_adjacency(adj)

        train_mask, val_mask, test_mask = create_masks(num_nodes, labels, num_classes=len(unique_classes))

        return {
            'x': features,
            'y': labels,
            'adj': adj_norm,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'num_features': features.shape[1],
            'num_classes': len(unique_classes),
            'num_nodes': num_nodes
        }


def create_masks(num_nodes, labels, num_classes=7, train_per_class=20, val_num=500, test_num=1000, seed=42):
    """
    Create train/val/test masks following the standard split from the paper.

    Standard split for Cora (fixed with seed for reproducibility):
    - 20 nodes per class for training (140 total for 7 classes)
    - 500 nodes for validation
    - 1000 nodes for testing

    Args:
        num_nodes: Total number of nodes
        labels: Node labels tensor [N]
        num_classes: Number of classes
        train_per_class: Number of training samples per class
        val_num: Number of validation samples
        test_num: Number of test samples
        seed: Random seed for reproducible splits

    Returns:
        train_mask, val_mask, test_mask: Boolean tensors [N]
    """
    # Set seed for reproducible mask generation
    torch.manual_seed(seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Training set: 20 nodes per class
    for c in range(num_classes):
        class_indices = (labels == c).nonzero(as_tuple=True)[0]
        # Randomly shuffle and take first 20 (with fixed seed for reproducibility)
        perm = torch.randperm(len(class_indices))
        train_indices = class_indices[perm[:train_per_class]]
        train_mask[train_indices] = True

    # Remaining nodes for val and test
    remaining = (~train_mask).nonzero(as_tuple=True)[0]
    perm = torch.randperm(len(remaining))

    val_indices = remaining[perm[:val_num]]
    test_indices = remaining[perm[val_num:val_num + test_num]]

    val_mask[val_indices] = True
    test_mask[test_indices] = True

    print(f"Dataset split:")
    print(f"  Train: {train_mask.sum().item()} nodes ({train_per_class} per class × {num_classes} classes)")
    print(f"  Val: {val_mask.sum().item()} nodes")
    print(f"  Test: {test_mask.sum().item()} nodes")
    print(f"  Unlabeled: {(num_nodes - train_mask.sum() - val_mask.sum() - test_mask.sum()).item()} nodes")

    return train_mask, val_mask, test_mask


if __name__ == '__main__':
    """Test data loader."""
    print("=" * 70)
    print("Testing CoraDataLoader")
    print("=" * 70)

    # Test loading Cora dataset (uses default root=None to auto-locate project directory)
    loader = CoraDataLoader()
    data = loader.load()

    print("\n" + "=" * 70)
    print("Cora Dataset Loaded Successfully!")
    print("=" * 70)

    print(f"\nData Shapes:")
    print(f"  Node features (x):          {data['x'].shape}")
    print(f"  Labels (y):                 {data['y'].shape}")
    print(f"  Normalized adjacency (adj): {data['adj'].shape}")

    print(f"\nDataset Statistics:")
    print(f"  Number of nodes:            {data['num_nodes']}")
    print(f"  Number of features:         {data['num_features']}")
    print(f"  Number of classes:          {data['num_classes']}")

    print(f"\nMask Statistics:")
    print(f"  Train mask:                 {data['train_mask'].shape} | True: {data['train_mask'].sum().item()}")
    print(f"  Val mask:                   {data['val_mask'].shape} | True: {data['val_mask'].sum().item()}")
    print(f"  Test mask:                  {data['test_mask'].shape} | True: {data['test_mask'].sum().item()}")

    print(f"\nTensor Value Checks:")
    print(f"  Feature matrix dtype:       {data['x'].dtype}")
    print(f"  Feature range:              [{data['x'].min():.4f}, {data['x'].max():.4f}]")
    print(f"  Label dtype:                {data['y'].dtype}")
    print(f"  Label range:                [{data['y'].min()}, {data['y'].max()}]")
    print(f"  Adjacency dtype:            {data['adj'].dtype}")
    print(f"  Adjacency range:            [{data['adj'].min():.4f}, {data['adj'].max():.4f}]")

    # Verify normalization is symmetric
    is_symmetric = torch.allclose(data['adj'], data['adj'].T, atol=1e-6)
    print(f"  Adjacency symmetric:        {is_symmetric}")

    # Verify feature normalization (row norms should be ~1)
    row_norms = torch.norm(data['x'], p=2, dim=1)
    print(f"  Feature row norms (L2):     mean={row_norms.mean():.4f}, std={row_norms.std():.6f}")

    # Verify label distribution in training set
    train_labels = data['y'][data['train_mask']]
    unique, counts = torch.unique(train_labels, return_counts=True)
    print(f"\nTraining set label distribution:")
    for cls, count in zip(unique.tolist(), counts.tolist()):
        print(f"  Class {cls}: {count} samples")

    print("\n" + "=" * 70)
    print("All tests passed! Data loader is working correctly.")
    print("=" * 70)
