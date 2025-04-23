# Urban Street Network Morphology Classification Through SGBNet and Multi-Model Fusion

This repository contains the code used for the paper *"Urban Street Network Morphology Classification Through Street-Block Based Graph Neural Networks and Multi-model Fusion"*. The proposed method integrates a custom Graph Neural Network (SGBNet), Convolutional Neural Networks (CNN) using ResNet-34, and a Multi-Layer Perceptron (MLP) for urban street network classification.

## Code Overview

The code includes the following components:

1. **SGBNet**: A custom Graph Neural Network designed to represent urban street networks as graphs, with street blocks as nodes and geometric features as node attributes.
2. **CNN**: A convolutional neural network to extract visual features from images of urban street networks.
3. **MLP**: A multi-layer perceptron for computing global descriptors from urban network data.
4. **Multi-Model Fusion**: The fusion of features from the SGBNet, CNN, and MLP models for improved classification performance.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/liuyangliuyangliuyang/Urban-Street-Network-Morphology-Classification-Through-SGBNet-and-Fusion-Models
