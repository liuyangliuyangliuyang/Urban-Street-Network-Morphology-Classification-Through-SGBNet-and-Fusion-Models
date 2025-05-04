
# Urban Street Network Morphology Classification Through SGBNet and Multi-Model Fusion

This repository contains the code used for the paper *"Urban Street Network Morphology Classification Through Street-Block Based Graph Neural Networks and Multi-model Fusion"*. The proposed method integrates a custom Graph Neural Network (SGBNet), Convolutional Neural Networks (CNN) using ResNet-34, and a Multi-Layer Perceptron (MLP) for urban street network classification.

## Code Overview

The code includes the following components:

1. **SGBNet**: A custom Graph Neural Network designed to represent urban street networks as graphs, with street blocks as nodes and geometric features as node attributes.
2. **CNN**: A convolutional neural network to extract visual features from images of urban street networks.
3. **MLP**: A multi-layer perceptron for computing global descriptors from urban network data.
4. **Multi-Model Fusion**: The fusion of features from the SGBNet, CNN, and MLP models for improved classification performance.

## Directory Structure

.
# Project Directory Structure

## dataset/ # Data loading and preprocessing
- `init.py`
- `CombinedDataset.py` # Combined dataset definition and handling
- `DataSource.py` # Data source loading and handling
- `load_splits_from_h5py.py` # Dataset split loading from HDF5 files
- `IndexedDataset.py` # Dataset wrapper to return data with indices
- `dataset_utils.py` # Helper functions for dataset processing

## feature_processing/ # Feature processing and transformation
- `init.py`
- `feature_extraction.py` # Extracting features from datasets
- `feature_transform.py` # Feature transformation (scaling, etc.)
- `feature_utils.py` # Utility functions for feature handling
- `transform_features.py` # Transform node features (GNN) and global features

## models/ # Model definitions
- `init.py`
- `cnn.py` # CNN model components
- `gnn.py` # GNN model components
- `fusion.py` # Fusion of different model features
- `dynamic_model.py` # Dynamic model combining CNN, GNN, and global features

## trainer/ # Training and evaluation functions
- `init.py`
- `train.py` # Training loop function
- `evaluate.py` # Evaluation function for validation and testing
- `test.py` # Testing loop and final evaluation

## utils/ # Utility functions
- `init.py`
- `save_results.py` # Saving results and logs (training costs, metrics)
- `metrics.py` # Calculating performance metrics (F1, accuracy, etc.)
- `cost_utils.py` # Helper functions for saving cost metrics (training time, model size)
- `plot_utils.py` # Utility functions for plotting and visualizations

## config/ # Configuration files
- `config.py` # Configuration for all models (CNN, GNN, etc.)

## main.py # Main entry point for running the training and evaluation pipeline




## Installation

### Clone the repository

To clone the repository, use the following command:

```bash
git clone https://github.com/yourusername/Urban-Street-Network-Morphology-Classification-Through-SGBNet-and-Fusion-Models.git

### Install dependencies
To install the necessary dependencies, run:

pip install -r requirements.txt
The requirements.txt file contains the required libraries for running the models:



torch
torchvision
torch-geometric
scikit-learn
h5py
networkx
matplotlib
numpy
Usage

Configuration
The configuration for the models is located in config/config.py. This file allows you to specify the models (CNN, GNN, etc.), their parameters, and how to combine them. Below is an example configuration:

python

possible_models = {
    'cnn': {'type': 'image', 'columns': ['images'], 'features': 512},
    'gnn0': {
        'type': 'graph0',
        'columns': ['nx_list'],
        'input_feature_size': 8,
        'output_feature_size': 256,
        'params': {
            'model_embedding_size': 128,
            'model_attention_heads': 1,
            'model_layers': 2,
            'model_dropout_rate': 0.2,
            'model_top_k_ratio': 0.5,
            'model_top_k_every_n': 1,
            'model_dense_neurons': 256,
            'model_edge_dim': 2
        }
    },
    'label': {'type': 'label', 'columns': ['label0']},
    # Add other model configurations as needed
}
Dataset Loading
The dataset/ directory contains the necessary functions to load and preprocess your data. You can load data like this:


from dataset import DataSource

# Load graph data from file
data_source = DataSource(file_path='data/urban-street-network.h5', data_type='graph0', columns=['nx_list'])
data = data_source.load_data(idx=0)
Model Training and Evaluation
To start the training and evaluation process, run the main.py script:


python main.py
This script will automatically load the data, initialize the model, and begin the training process. You can adjust the configurations in config/config.py to modify the model or dataset.

Feature Processing
The feature_processing/ directory includes all functions for extracting and transforming features. You can transform features as follows:


from feature_processing import transform_node_features, transform_features

train_dataset, val_dataset, test_dataset = transform_node_features(train_dataset, val_dataset, test_dataset)
Model Definition
The models (CNN, GNN, etc.) are located in the models/ directory. Below is an example of how to define and use a dynamic model:


from models import DynamicModel

model = DynamicModel(config=possible_models, num_classes=6)
Detailed Components
Dataset (dataset/)
DataSource.py: Handles loading data from various sources (images, graphs, labels).

load_image(idx): Loads image data.

load_graph0(idx), load_graph1(idx): Loads graph data of type 0 and 1.

load_labels(idx): Loads labels for each data sample.

CombinedDataset.py: Combines multiple data sources into a single dataset.

__getitem__(self, idx): Returns data from all sources at the specified index.

__len__(self): Returns the length of the dataset.

load_splits_from_h5py.py: Loads dataset splits (train, validation, test) from an HDF5 file.

load_splits_from_h5py(h5_file_path, key): Loads dataset splits from the specified key in the H5 file.

Feature Processing (feature_processing/)
feature_extraction.py: Extracts features from the dataset.

extract_node_features(dataset): Extracts node features for GNNs.

extract_global_features(dataset): Extracts global features.

feature_transform.py: Transforms features (e.g., scaling, normalization).

transform_node_features(train_dataset, val_dataset, test_dataset): Transforms node features for GNNs.

transform_features(train_dataset, val_dataset, test_dataset): Transforms global features.

Models (models/)
cnn.py: Defines CNN-based models for image data.

ModifiedResNet34: A modified ResNet34 model for image feature extraction.

gnn.py: Defines GNN models for graph-based data.

CustomGNN: A custom GNN model with configurable layers and pooling mechanisms.

fusion.py: Defines a model that fuses features from different models.

FusionLayer: Combines CNN and GNN outputs for final classification.

dynamic_model.py: A dynamic model that integrates CNN, GNN, and global models.

DynamicModel: Main model that integrates CNN, GNN, and global models into a unified framework.

Trainer (trainer/)
train.py: Implements the training loop.

train(model, train_loader, optimizers, criterion, device): Trains the model for one epoch.

evaluate.py: Implements the evaluation loop.

evaluate(model, loader, criterion, device): Evaluates the model on validation/test data.

test.py: Performs the final evaluation after training.

evaluate1(model, loader, criterion, device): Evaluates the model and saves results.

Utils (utils/)
save_results.py: Handles saving results, such as training cost and performance metrics.

save_cost(training_time, model_parameters, split_seed, root_result_path): Saves training time and model parameters.

metrics.py: Calculates performance metrics such as F1 score, accuracy, precision, and recall.

calculate_metrics(y_pred, y_true): Calculates overall metrics like F1 score, accuracy, and confusion matrix.

cost_utils.py: Contains helper functions to save training cost and other metrics.

Example Usage
To run the training and evaluation pipeline, execute main.py:


python main.py
You can modify the configurations and models by editing the config/config.py file.



This is the complete **README.md** in **Markdown** format, which includes the entire directory structure and all relevant details. You can copy this entire content and paste it into your GitHub repository without worrying about formatting issues.







