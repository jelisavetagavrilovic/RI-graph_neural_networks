# Node Classification with Graph Neural Networks

This project focuses on node classification in ego networks from the Facebook social network dataset using Graph Neural Networks (GNNs). We compare the performance of two GNN architectures: Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT).

## Introduction

Ego networks are structures within social networks centered around a single individual node (the "ego") and their connections with other users. This project aims to classify users based on the ego networks they belong to, using interaction data and anonymized attributes.

## Dataset

The dataset is from the [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/ego-Facebook.html), specifically the Facebook dataset. It is represented as a graph where:
- Nodes represent users (total: 4039)
- Edges represent friendships between users (total: 88,234)
- Each user is described by a set of binary attributes (2255 in total, anonymized).

## Problem Definition

The goal is to predict which ego network a user belongs to based on the user’s interactions and attributes. Users can belong to one, multiple, or no ego networks.

## Solution Overview

We implemented two types of Graph Neural Networks to solve the node classification problem:
1. **Graph Convolutional Networks (GCN)**: 
   - GCN collects information from a node's neighbors and updates the node’s representation based on this aggregated information.
   - We used a 2-layer GCN with a dropout layer for regularization.

2. **Graph Attention Networks (GAT)**: 
   - GATs, on the other hand, use an attention mechanism to assign different weights to neighbors, allowing the model to focus on the most relevant ones. It also employs multi-head attention for richer node representations.
   - We implemented a 2-layer GAT model with multiple attention heads and a dropout layer.

## Results

We found that the GAT model performs better than the GCN in terms of F1-Score, particularly due to the imbalance in the target classes. However, this comes with the trade-off of significantly longer training times.

| Model  | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| GCN    | 95.57%   | 0.7121    | 0.9727 | 0.7869   |
| GAT    | 95.97%   | 0.7767    | 0.9756 | 0.8428   |

## How to Run

1. Clone this repository.
2. Install the required dependencies:
```bash
  pip install pandas networkx matplotlib numpy seaborn collections pickle torch torchmetrics torch_geometric sklearn itertools
```
3. Run Python scripts.
