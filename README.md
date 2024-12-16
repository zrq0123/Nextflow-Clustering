# Nextflow-Clustering

# Overview
In this study, we deploy nextflow for clustering single-cell RNA-seq data. This clustering method CAKE is a novel and scalable self-supervised clustering method, which consists of a contrastive learning model with a mixture neighborhood augmentation for cell representation learning, and a self-Knowledge Distiller model for refinement of clustering results. These designs provide more condensed and cluster-friendly cell representations and improve the clustering performance in term of accuracy and robustness.

# Usage

```bash
nextflow run cluster.nf --File_Type Muraro

