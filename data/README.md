# Datasets Directory

This directory contains the citation network datasets used for GCN training.

## Datasets

### 1. Cora (Primary Dataset)
- **Description**: Citation network of machine learning papers
- **Nodes**: 2,708 papers
- **Edges**: 5,429 citation links
- **Features**: 1,433 (binary word vectors)
- **Classes**: 7 (paper categories)
- **Download**: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz

### 2. CiteSeer (Alternative)
- **Description**: Citation network of computer science papers
- **Nodes**: 3,327 papers
- **Edges**: 4,732 citation links
- **Features**: 3,703 (binary word vectors)
- **Classes**: 6 (paper categories)
- **Download**: https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz

### 3. PubMed (Not recommended for from-scratch implementation)
- **Description**: Citation network of diabetes-related papers
- **Nodes**: 19,717 papers
- **Edges**: 44,338 citation links
- **Features**: 500 (TF-IDF weighted vectors)
- **Classes**: 3 (paper categories)

## Dataset Format

Each dataset typically comes with:
- `*.content`: Node features and labels
- `*.cites`: Edge list (citation relationships)

## Data Loading

The `src/data_loader.py` module handles automatic downloading and parsing.
Simply run the training script - it will download the dataset on first use.

## References

1. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR 2017.
2. Sen, P., et al. (2008). Collective Classification in Network Data. AI Magazine.
