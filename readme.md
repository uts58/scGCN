# An Exploration of Integrative Analysis of Simultaneously Profiled scHi-C and scRNA-seq Data Using Graph Convolutional Network
#### Paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11129598)
---

**scGCN** is a PyTorch Geometric-based framework for integrative analysis of simultaneously profiled single-cell Hi-C (scHi-C) and single-cell RNA-seq (scRNA-seq) data using unsupervised Graph Convolutional Networks (GCNs). It transforms chromatin interactions and gene expression into graphs to uncover relationships between 3D genome architecture and gene expression in single cells.

---

## 📌 Key Features

- Integration of scHi-C and scRNA-seq data at single-cell resolution
- Graph construction per chromosome using 50kb genomic bins
- Two GCN-based models:
  - `ModelDeep`: uses node features (UMI counts)
  - `ModelDeepNoFeatures`: learns node embeddings from graph topology
- Unsupervised training with variance-based loss
- Embedding extraction and dimensionality reduction via UMAP
- Clustering and evaluation using ARI and Silhouette scores

---

## 🧪 Datasets

scGCN is tested using three publicly available mouse single-cell multi-omics datasets:

- **GSE223917** – Mouse brain and embryo
- **GSE211395** – Mouse embryonic stem cells (2i vs. serum media)
- **GSE239969** – Mouse olfactory epithelium (two strains)

Raw data: [NCBI GEO](https://www.ncbi.nlm.nih.gov/)

---

## 🔧 Installation

```bash
git clone https://github.com/uts58/scGCN.git
cd scGCN
pip install -r requirements.txt
```

---

## 📊 Results Overview

- **Silhouette Scores**: When using only scHi-C data, the model achieved a median Silhouette score of approximately 0.6, indicating strong separation between clusters.
- **Adjusted Rand Index (ARI)**: ARI scores were generally low (median near 0), suggesting that while the model finds structured clusters, they often do not align with known biological labels—potentially revealing novel cellular states.
- **Effect of Integration**: Adding scRNA-seq data led to improved clustering in certain chromosomes, but also introduced noise in others, reducing overall cluster cohesion. This reflects the complex and variable nature of multi-omics integration.

---

## 📈 Evaluation Metrics

- **Adjusted Rand Index (ARI)**  
  Quantifies similarity between predicted clusters and ground truth labels. Values range from -1 to 1, where 1 indicates perfect agreement.

- **Silhouette Score**  
  Measures how well each sample fits within its assigned cluster. Higher scores indicate more distinct and well-separated clusters.

These metrics are evaluated across all chromosomes and at different embedding dimensions to provide a comprehensive performance analysis.

---

## 🤝 Acknowledgments
This work was supported by the Center for Computationally Assisted Science and Technology (CCAST) at North Dakota State University, enabled by NSF MRI Award No. 2019077.

