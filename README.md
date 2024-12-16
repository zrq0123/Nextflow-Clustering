# Nextflow-Clustering

## Overview
In this study, we deploy nextflow for clustering single-cell RNA-seq data. This clustering method CAKE is a novel and scalable self-supervised clustering method, which consists of a contrastive learning model with a mixture neighborhood augmentation for cell representation learning, and a self-Knowledge Distiller model for refinement of clustering results. These designs provide more condensed and cluster-friendly cell representations and improve the clustering performance in term of accuracy and robustness.

## Installation
The versions of some packages used in the experiment are as follows:
nextflow version 24.10.2.5932 \
anndata==0.9.2 \
h5py==3.11.0 \
hnswlib==0.8.0 \
numba==0.57.1 \
numpy==1.24.3 \
pandas==1.3.5 \
python==3.8.16 \
pytorch==1.13.1 \
pytorch-cuda==11.7 \
scanpy==1.9.2 \
scikit-learn==1.2.2 \
scipy==1.9.3 \
seaborn==0.12.2 


# Usage

```bash
nextflow run cluster.nf --File_Type Muraro

