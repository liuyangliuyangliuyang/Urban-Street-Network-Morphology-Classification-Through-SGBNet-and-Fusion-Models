import os
import pickle

import sklearn
import torch
import torch.nn.functional as F
import torch_geometric
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    classification_report, average_precision_score, roc_curve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, TopKPooling, Linear
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image
from torch_geometric.utils import from_networkx
import numpy as np
import pandas as pd
import networkx as nx
import h5py
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric.nn.conv.rgcn_conv

from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling, GATConv, GCNConv, ChebConv, ClusterGCNConv, DenseSAGEConv, \
    DenseGCNConv, RGCNConv, dense_diff_pool, SAGPooling, GIN, GINConv
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from torch_geometric.nn import GNNExplainer
from torch_geometric.nn import GINEConv
from torchvision.models import resnet34
from torch_geometric.nn import GAT
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from config import HYPERPARAMETERS, BEST_PARAMETERS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from torch_geometric.data import Data
import networkx as nx
import torch
from torchvision.transforms import ToTensor

from torch.utils.data import Dataset, DataLoader

# %% imports
import csv
import pickle
from itertools import chain
from turtle import goto
import sklearn.metrics
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler, SubsetRandomSampler
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, \
    classification_report, average_precision_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize

import numpy as np
from tqdm import tqdm
# from dataset_featurizer import MoleculeDataset
from model import GNN
from model_test import GNN_test
from GCNmodel import GCN
# import mlflow.pytorch
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd

import os
import networkx as nx
import mydataset_test
import random
from sklearn.metrics import f1_score

combined_features_storage = []


def save_activation(name):
    def hook(model, input, output):
        global combined_features_storage
        combined_features_storage.append(output.detach().cpu().numpy())  # Append features of each batch

    return hook


def post_process_features():
    global combined_features_storage
    combined_features = np.concatenate(combined_features_storage, axis=0)  # Concatenate all batch features
    combined_features_storage = []  # Reset for the next use
    return combined_features


def post_process_features1():
    global combined_features_storage
    # print(combined_features_storage)
    print(f"Before sorting: {combined_features_storage[:10]}")  # Print first 10 entries for debugging
    combined_features_storage.sort(key=lambda x: x[1])  # Sort by index
    print(f"Before sorting: {combined_features_storage[:10]}")  # Print first 10 entries for debugging
    combined_features, indices = zip(*combined_features_storage)
    combined_features = np.concatenate(combined_features, axis=0)
    combined_features_storage = []
    return combined_features


def post_process_features2():
    global combined_features_storage

    combined_features = np.concatenate(combined_features_storage, axis=0)  # Concatenate all batch features

    print(combined_features_storage)
    # combined_features, indices = zip(*combined_features_storage)
    # # combined_features = np.vstack(combined_features)
    # combined_features = np.concatenate(combined_features, axis=0)
    combined_features_storage = []  # Reset for the next use
    return combined_features, indices


class IndexedDataset0(Dataset):
    """Dataset wrapper to return the index of the item along with the data."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]
        return data, index

    def __len__(self):
        return len(self.dataset)


class IndexedDataset(Dataset):
    """Dataset wrapper to return the index of the item along with the data."""

    def __init__(self, dataset, original_indices):
        self.dataset = dataset
        self.original_indices = original_indices

    def __getitem__(self, index):
        data = self.dataset[index]
        original_index = self.original_indices[index]
        return data, original_index

    def __len__(self):
        return len(self.dataset)


def collate_data_with_index(batch):
    batched_data, indices = zip(*batch)
    return collate_data(batched_data), indices  # Assuming collate_data is your existing function


import time
import csv



def save_micro_average_precision_recall_curve(y_true, y_scores, classes, root_csv_file_path):
    num_classes = len(classes)
    plt.figure(figsize=(10, 8))

    # Binarize the true labels
    # y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])

    # Initialize variables for micro-average precision-recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    auc_values = []

    for i in range(num_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        auc_prc = auc(recall[i], precision[i])
        auc_values.append(auc_prc)

        # Save precision-recall curve data to one CSV file for each class
        prc_curve_data = list(zip(recall[i], precision[i]))
        prc_curve_filename = os.path.join(root_csv_file_path, f'prc_curve_class_{i}.csv')

        with open(prc_curve_filename, 'w', newline='') as prc_csv:
            csv_writer = csv.writer(prc_csv)
            csv_writer.writerow(['Recall', 'Precision'])
            csv_writer.writerows(prc_curve_data)

    # Micro-average for precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_scores.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin.ravel(), y_scores.ravel())

    # Save micro-average precision-recall curve data to a separate CSV file
    prc_curve_data_micro = list(zip(recall["micro"], precision["micro"]))
    prc_curve_filename_micro = os.path.join(root_csv_file_path, 'prc_curve_micro_average.csv')

    with open(prc_curve_filename_micro, 'w', newline='') as prc_micro_csv:
        csv_writer_micro = csv.writer(prc_micro_csv)
        csv_writer_micro.writerow(['Recall', 'Precision'])
        csv_writer_micro.writerows(prc_curve_data_micro)

    # Save all AUC values to one CSV file
    auc_values_filename = os.path.join(root_csv_file_path, 'auc_values_precision_recall.csv')
    with open(auc_values_filename, 'w', newline='') as auc_csv:
        csv_writer = csv.writer(auc_csv)
        csv_writer.writerow(['Class', 'AUC'])
        # Write individual AUC values for each class
        for c, auc_val in zip(classes, auc_values):
            csv_writer.writerow([c, auc_val])

        # Write micro-average AUC value
        csv_writer.writerow(['Micro-average', average_precision["micro"]])
    print('aupr-saved')


def save_micro_average_roc_curve(y_true, y_scores, classes, root_csv_file_path):
    num_classes = len(classes)
    plt.figure(figsize=(12, 8))

    # Binarize the true labels
    # y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])

    # Initialize variables for micro-average ROC curve
    all_fpr = []
    all_tpr = []
    auc_values = []

    # For each class
    for i in range(num_classes):
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        auc_roc = roc_auc_score(y_true_bin[:, i], y_scores[:, i])
        # Collect data for micro-average
        all_fpr.append(fpr)
        all_tpr.append(tpr)
        auc_values.append(auc_roc)

    # Micro-average for ROC curve
    micro_fpr, micro_tpr, _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
    micro_auc_roc = roc_auc_score(y_true_bin.ravel(), y_scores.ravel())

    # Save all ROC curve data to one CSV file
    roc_curve_data = list(zip(*all_fpr, *all_tpr))
    roc_curve_data.append(micro_fpr)
    roc_curve_data.append(micro_tpr)

    # Save each ROC curve data to separate CSV files
    for i, (fpr, tpr) in enumerate(zip(all_fpr, all_tpr), start=1):
        curve_data = list(zip(fpr, tpr))
        curve_filename = os.path.join(root_csv_file_path, f'roc_curve_class_{i}.csv')

        with open(curve_filename, 'w', newline='') as curve_csv:
            csv_writer = csv.writer(curve_csv)
            csv_writer.writerow(['FPR', 'TPR'])
            csv_writer.writerows(curve_data)

    # Save micro-average ROC curve data to a separate CSV file
    micro_curve_data = list(zip(micro_fpr, micro_tpr))
    micro_curve_filename = os.path.join(root_csv_file_path, 'roc_curve_micro_average.csv')

    with open(micro_curve_filename, 'w', newline='') as micro_curve_csv:
        csv_writer = csv.writer(micro_curve_csv)
        csv_writer.writerow(['Micro FPR', 'Micro TPR'])
        csv_writer.writerows(micro_curve_data)

    # Save all AUC values to one CSV file
    auc_values_filename = os.path.join(root_csv_file_path, 'auc_values.csv')
    with open(auc_values_filename, 'w', newline='') as auc_csv:
        csv_writer = csv.writer(auc_csv)
        csv_writer.writerow(['Class', 'AUC'])
        # Write individual AUC values for each class
        for c, auc_val in zip(classes, auc_values):
            csv_writer.writerow([c, auc_val])

        # Write micro-average AUC value
        csv_writer.writerow(['Micro-average', micro_auc_roc])
    print('roc-saved')


def save_classification_metrics(y_true, y_pred, average_type, type, seed, root_result_path):
    # Calculate F1 score
    f1_score_value = f1_score(y_true, y_pred, average=average_type)
    accuracy_score_value = accuracy_score(y_true, y_pred)

    # Calculate precision
    precision_value = precision_score(y_true, y_pred, average=average_type)

    # Calculate recall
    recall_value = recall_score(y_true, y_pred, average=average_type)

    # Save each metric to separate CSV files
    metrics_data = {
        'F1_Score': f1_score_value,
        'Accuracy_Score': accuracy_score_value,
        'Precision': precision_value,
        'Recall': recall_value
    }

    metrics_filename = os.path.join(root_result_path, f'classification_metrics_{type}_seed_{seed}.csv')

    with open(metrics_filename, 'w', newline='') as metrics_csv:
        csv_writer = csv.writer(metrics_csv)
        csv_writer.writerow(['Metric', 'Value'])
        csv_writer.writerows(metrics_data.items())
    print('metrics-saved')


def convert_multigraph_to_simple(multigraph):
    # Create an empty simple graph
    simple_graph = nx.Graph()

    for u, v, data in multigraph.edges(data=True):
        if simple_graph.has_edge(u, v):
            # Edge already exists, so we take the mean of the 'angle' attribute
            simple_graph[u][v]['angle'] = np.mean([simple_graph[u][v]['angle'], data.get('angle', 0)])
        else:
            # Otherwise, add a new edge with the 'angle' attribute
            simple_graph.add_edge(u, v, angle=data.get('angle', 0))

    return simple_graph


import h5py
import torch
from torch.utils.data import Dataset


class DataSource:
    def __init__(self, file_path, data_type, columns, label_column='label0'):
        self.file = h5py.File(file_path, 'r+')
        self.data_type = data_type
        self.columns = columns
        self.label_column = label_column  # Optional, for integrating labels with graph data

    def load_data(self, idx):
        if self.data_type == 'image':
            return self.load_image(idx)
        elif self.data_type == 'graph0':
            return self.load_graph0(idx)
        elif self.data_type == 'graph1':
            return self.load_graph1(idx)
        elif self.data_type == 'graph2':
            return self.load_graph2(idx)
        elif self.data_type == 'global':
            return self.load_global_features(idx)
        elif self.data_type == 'global0':
            return self.load_global_features(idx)
        elif self.data_type == 'label':
            return self.load_labels(idx)
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def load_image(self, idx):
        # Assuming images are stored in HWC format and need to be converted to CHW for PyTorch
        image_data = torch.tensor(self.file['images'][idx], dtype=torch.float)
        image_data = image_data.permute(2, 0, 1)  # Permute from HWC to CHW
        return image_data / 255.0  # Normalize to [0, 1]

    def load_graph0(self, idx):
        graphs = []
        label = self.load_labels(idx) if self.label_column else None
        for column in self.columns:
            graph_data = pickle.loads(self.file[column][idx].tobytes())
            node_features, edge_index, edge_attr = self.process_graph0(graph_data)
            # Include the label in the graph data if present
            graph_obj = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
            graphs.append(graph_obj)
        return graphs

    def load_graph1(self, idx):
        graphs = []
        label = self.load_labels(idx) if self.label_column else None
        for column in self.columns:
            graph_data = pickle.loads(self.file[column][idx].tobytes())
            node_features, edge_index, edge_attr = self.process_graph1(graph_data)
            # Include the label in the graph data if present
            graph_obj = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
            graphs.append(graph_obj)
        return graphs

    def load_graph2(self,idx):
        graphs = []
        label = self.load_labels(idx) if self.label_column else None
        for column in self.columns:
            graph_data = pickle.loads(self.file[column][idx].tobytes())
            node_features, edge_index, edge_attr = self.process_graph2(graph_data)
            # Include the label in the graph data if present
            graph_obj = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=label)
            graphs.append(graph_obj)
        return graphs

    def process_graph0(self, G):
        # Convert G to node features, edge index, and edge attributes
        node_feats = self._get_node_features0(G)
        edge_feats = self._get_edge_features0(G)
        edge_index = self._get_adjacency_info0(G)

        return node_feats, edge_index, edge_feats

    def process_graph1(self, G):
        # Convert G to node features, edge index, and edge attributes
        node_feats = self._get_node_features1(G)
        edge_feats = self._get_edge_features1(G)
        edge_index = self._get_adjacency_info1(G)

        return node_feats, edge_index, edge_feats

    def process_graph2(self, G):
        # Convert G to node features, edge index, and edge attributes
        node_feats = self._get_node_features2(G)
        edge_feats = self._get_edge_features2(G)
        edge_index = self._get_adjacency_info2(G)

        return node_feats, edge_index, edge_feats
    def load_labels(self, idx):
        # Labels are loaded only if a label column has been specified
        if self.label_column:
            label = torch.tensor(self.file[self.label_column][idx], dtype=torch.long)
            return label
        return None

    @staticmethod
    def _get_node_features0(G):
        all_node_feats = []
        feature_keys = ['area', 'circuity', 'concavity', 'number_of_linestrings', 'formfacter', 'rectanglarity',
                        'elongation', 'degree']
        for node in G.nodes():
            node_feats = []
            for key in feature_keys:
                if key == 'area':  # Scale 'area' feature by dividing by 10000
                    area_value = G.nodes[node].get(key, 0) / 10000
                    node_feats.append(area_value)
                else:
                    node_feats.append(G.nodes[node].get(key, 0))
            all_node_feats.append(node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    @staticmethod
    def _get_node_features1(G):
        all_node_feats = []
        feature_keys = ['circuity', 'sinuosity', 'degree', 'type']
        for node in G.nodes():
            node_feats = []
            for key in feature_keys:
                node_feats.append(G.nodes[node].get(key, 0))
            all_node_feats.append(node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    @staticmethod
    def _get_node_features2(G):
        all_node_feats = []
        feature_keys = ['degree','betweenness_centrality', 'closeness_centrality', 'clustering', 'pagerank']
        for node in G.nodes():
            node_feats = []
            for key in feature_keys:
                node_feats.append(G.nodes[node].get(key, 0))
            all_node_feats.append(node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    @staticmethod
    def _get_edge_features0(G):
        all_edge_feats = []
        for edge in G.edges():
            edge_feats = []
            # Original logic for 'nx_list'
            edge_feats.append(0)  # Placeholder; adjust as needed
            edge_feats.append(0)  # Placeholder; adjust as needed
            all_edge_feats += [edge_feats, edge_feats]  # Consider if duplication is needed based on graph structure
        all_edge_feats = np.asarray(all_edge_feats)
        # print(all_edge_feats.shape)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    @staticmethod
    def _get_edge_features1(G):
        all_edge_feats = []
        for edge in G.edges():
            edge_feats = []
            # Append 'angle' to edge features
            angle = G.edges[edge[0], edge[1], 0]["angle"]
            edge_feats.append(angle)
            # edge_feats.append(angle_value)
            all_edge_feats += [edge_feats, edge_feats]  # Consider if duplication is needed based on graph structure
        all_edge_feats = np.asarray(all_edge_feats)
        # print(all_edge_feats.shape)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    @staticmethod
    def _get_edge_features2(G):
        all_edge_feats = []
        for edge in G.edges():
            edge_feats = []
            # Original logic for 'nx_list'
            edge_feats.append(0)  # Placeholder; adjust as needed
            edge_feats.append(0)  # Placeholder; adjust as needed
            all_edge_feats += [edge_feats, edge_feats]  # Consider if duplication is needed based on graph structure
        all_edge_feats = np.asarray(all_edge_feats)
        # print(all_edge_feats.shape)
        return torch.tensor(all_edge_feats, dtype=torch.float)

    @staticmethod
    def _get_adjacency_info0(G):
        """
        We could also use rdmolops.GetAdjacencyMatrix(mol)
        but we want to be sure that the order of the indices
        matches the order of the edge features
        """
        G = nx.convert_node_labels_to_integers(G)
        G = G.to_directed() if not nx.is_directed(G) else G
        if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
            edges = list(G.edges(keys=False))
        else:
            edges = list(G.edges)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # print(edge_index.shape)
        return edge_index

    @staticmethod
    def _get_adjacency_info1(G):
        G = nx.convert_node_labels_to_integers(G)
        # If the graph is a MultiGraph, convert it to a simple graph
        if isinstance(G, nx.MultiGraph):
            G = convert_multigraph_to_simple(G)
        # For an undirected graph, make sure to add both directions for each edge
        edges = []
        if not G.is_directed():
            for u, v in G.edges():
                edges.append((u, v))
                edges.append((v, u))
        else:
            edges = list(G.edges)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    @staticmethod
    def _get_adjacency_info2(G):
        G = nx.convert_node_labels_to_integers(G)
        # If the graph is a MultiGraph, convert it to a simple graph
        if isinstance(G, nx.MultiGraph):
            G = convert_multigraph_to_simple(G)
        # For an undirected graph, make sure to add both directions for each edge
        edges = []
        if not G.is_directed():
            for u, v in G.edges():
                edges.append((u, v))
                edges.append((v, u))
        else:
            edges = list(G.edges)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index

    def load_global_features(self, idx):
        # Combine global features from specified columns
        features = np.hstack([self.file[column][idx] for column in self.columns])
        return torch.tensor(features, dtype=torch.float)

    def close(self):
        self.file.close()


class CombinedDataset(Dataset):
    def __init__(self, file_path, config):
        """
        config is a dict specifying the data sources and their types, e.g.,
        {
            'cnn': {'type': 'image', 'columns': ['default']},
            'gnn': {'type': 'graph', 'columns': ['nx_list', 'dual_graph_nx_list']},
            'global': {'type': 'global', 'columns': ['features', 'hand_crafted']}
        }
        """
        self.data_sources = {}
        for key, source_config in config.items():
            self.data_sources[key] = DataSource(file_path, source_config['type'], source_config['columns'])

    def __len__(self):
        first_source = next(iter(self.data_sources.values()))
        # This assumes all data types are aligned and have the same length
        return len(first_source.file[first_source.columns[0]])

    def __getitem__(self, idx):
        data = {}
        for key, source in self.data_sources.items():
            data[key] = source.load_data(idx)
        # Unpack single-item lists for convenience
        for key in data:
            if isinstance(data[key], list) and len(data[key]) == 1:
                data[key] = data[key][0]
        return data

    def close(self):
        for source in self.data_sources.values():
            source.close()


import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx


def verify_dataset_content(data_sample):
    # Check and plot image data if present
    if 'cnn' in data_sample:
        # Assume 'cnn' contains the image data as a torch Tensor in CHW format
        image_data = data_sample['cnn']
        if image_data.shape[0] == 3:  # RGB image
            # Correct permutation from CHW (Channels, Height, Width) to HWC (Height, Width, Channels)
            image_data = image_data.permute(1, 2, 0)
        elif image_data.shape[0] == 1:  # Grayscale image
            # Remove channel dimension for grayscale image
            image_data = image_data.squeeze(0)

        # Convert tensor to numpy array if not already done
        if not isinstance(image_data, np.ndarray):
            image_data = image_data.numpy()

        plt.imshow(image_data)
        plt.title("Image Data")
        plt.colorbar()
        plt.show()

    # Check and plot graph data if present
    if 'gnn' in data_sample:
        for i, graph in enumerate(data_sample['gnn']):
            G = to_networkx(graph, to_undirected=True)
            plt.figure(i + 1)
            nx.draw(G, with_labels=True, node_color='skyblue', node_size=700, edge_color='k')
            plt.title(f"Graph Data {i + 1}")
            plt.show()

            # Print graph contents
            print(f"Graph {i + 1} Details:")
            print("Nodes:", G.nodes(data=True))
            print("Edges:", G.edges(data=True))
            if 'x' in graph:
                print("Node Features:", graph.x)
            if 'edge_index' in graph:
                print("Edge Indices:", graph.edge_index)
            if 'edge_attr' in graph:
                print("Edge Attributes:", graph.edge_attr)
            if 'y' in graph:
                print("Graph Label:", graph.y)

    # Print global data if present
    if 'global' in data_sample:
        print("Global Data:", data_sample['global'])

    # Print label if present
    if 'label' in data_sample:
        print("Label:", data_sample['label'])


def print_graph_attributes(g):
    # Check and print node attributes
    print("Node Attributes:")
    for node, attrs in g.nodes(data=True):
        print(f"Node {node}:")
        for attr, value in attrs.items():
            print(f"  {attr}: {value}")

    # Check and print edge attributes
    print("\nEdge Attributes:")
    for u, v, attrs in g.edges(data=True):
        print(f"Edge ({u}, {v}):")
        for attr, value in attrs.items():
            print(f"  {attr}: {value}")

from torch_geometric.data import Batch
def collate_data0(batch):
    """
    Custom collate function to handle multiple graph types and global features.
    Each item in the batch can contain multiple graph types and feature sets.
    """
    batched_data = {}

    # Example keys: 'gnn_nx_list', 'gnn_dual_graph_nx_list', 'global_features'
    keys = batch[0].keys()  # Assuming all items have the same structure

    for key in keys:
        if 'gnn0' in key:  # Graph data identified by 'gnn' prefix in key
            graph_data_list0 = [item[key] for item in batch]
            batched_data[key] = Batch.from_data_list(graph_data_list0)
        elif 'gnn1' in key:  # Graph data identified by 'gnn' prefix in key
            graph_data_list1 = [item[key] for item in batch]
            batched_data[key] = Batch.from_data_list(graph_data_list1)
        elif 'gnn2' in key:  # Graph data identified by 'gnn' prefix in key
            graph_data_list2 = [item[key] for item in batch]
            batched_data[key] = Batch.from_data_list(graph_data_list2)
        elif 'global' in key:  # Global features identified by 'global' prefix in key
            features_list = [item[key] for item in batch]
            batched_data[key] = torch.stack(features_list)
        elif 'global0' in key:  # Global features identified by 'global' prefix in key
            features_list = [item[key] for item in batch]
            batched_data[key] = torch.stack(features_list)
        elif 'cnn' in key:  # Image data identified by 'cnn' prefix in key
            images = torch.stack([item[key] for item in batch])
            batched_data[key] = images
        elif 'label' in key:  # Assuming 'label' key holds the labels
            labels = torch.tensor([item[key] for item in batch], dtype=torch.long)
            batched_data['label'] = labels

    return batched_data


def collate_data(batch):
    """
    Custom collate function to handle multiple graph types and global features.
    Each item in the batch can contain multiple graph types and feature sets.
    """
    batched_data = {}
    data_batch, indices = zip(*batch)  # Unzip the batch into data and indices
    # print(data_batch[0])
    # Example keys: 'gnn_nx_list', 'gnn_dual_graph_nx_list', 'global_features'
    keys = data_batch[0].keys()  # Assuming all items have the same structure
    # print(keys)
    # exit()
    for key in keys:
        if 'gnn0' in key:  # Graph data identified by 'gnn' prefix in key
            graph_data_list0 = [item[key] for item in data_batch]
            batched_data[key] = Batch.from_data_list(graph_data_list0)
        elif 'gnn1' in key:  # Graph data identified by 'gnn' prefix in key
            graph_data_list1 = [item[key] for item in data_batch]
            batched_data[key] = Batch.from_data_list(graph_data_list1)
        elif 'gnn2' in key:  # Graph data identified by 'gnn' prefix in key
            graph_data_list2 = [item[key] for item in data_batch]
            batched_data[key] = Batch.from_data_list(graph_data_list2)
        elif 'global' in key:  # Global features identified by 'global' prefix in key
            features_list = [item[key] for item in data_batch]
            batched_data[key] = torch.stack(features_list)
        elif 'global0' in key:  # Global features identified by 'global' prefix in key
            features_list = [item[key] for item in data_batch]
            batched_data[key] = torch.stack(features_list)
        elif 'cnn' in key:  # Image data identified by 'cnn' prefix in key
            images = torch.stack([item[key] for item in data_batch])
            batched_data[key] = images
        elif 'label' in key:  # Assuming 'label' key holds the labels
            labels = torch.tensor([item[key] for item in data_batch], dtype=torch.long)
            batched_data['label'] = labels

    return batched_data, indices  # Return both batched data and indices



import numpy as np
import sklearn.metrics
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, \
    classification_report


def calculate_metrics0(y_pred, y_true):
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(classification_report(y_true, y_pred))

# def calculate_metrics(y_pred, y_true,all_scores=None, save_results=False,split_seed = None,dataset_type = None, root_result_path=None):
#     average_type = "weighted"
#     print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
#     f1 = f1_score(y_true, y_pred, average="weighted")
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average="weighted")
#     recall = recall_score(y_true, y_pred, average="weighted")
#     print(f"F1 Score: {f1}")
#     print(f"Accuracy: {accuracy}")
#     print(f"Precision: {precision}")
#     print(f"Recall: {recall}")
#     print(classification_report(y_true, y_pred))
#     classes = ['0', '1', '2', '3', '4', '5']
#     # classes = ['0', '1', '2', '3']
#
#     if save_results == True:
#         seed_folder_path = os.path.join(root_result_path, split_seed)
#         if not os.path.exists(seed_folder_path):
#             os.makedirs(seed_folder_path)
#         datasettype_seed_folder_path = os.path.join(seed_folder_path, dataset_type)
#         if not os.path.exists(datasettype_seed_folder_path):
#             os.makedirs(datasettype_seed_folder_path)
#
#
#         save_micro_average_roc_curve(y_true, all_scores, classes, datasettype_seed_folder_path)
#         save_micro_average_precision_recall_curve(y_true, all_scores, classes, datasettype_seed_folder_path)
#         save_classification_metrics(y_true, y_pred, average_type, dataset_type, split_seed, datasettype_seed_folder_path)
#         print('save all result done')
from sklearn.metrics import classification_report
import csv


def calculate_metrics(y_pred, y_true, all_scores=None, save_results=False, split_seed=None, dataset_type=None,
                      root_result_path=None):
    # 计算整体的宏观指标
    average_type = "weighted"
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_true, y_pred)}")
    f1 = f1_score(y_true, y_pred, average="weighted")
    accuracy = accuracy_score(y_true, y_pred)  # 计算整体准确率
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    print(f"F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # 输出每个类别的详细分类报告
    class_report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))  # 打印到控制台

    # 计算每个类别的准确率（类的准确率 = 类别正确预测数 / 类别总数）
    accuracy_per_class = {}
    for class_label in class_report:
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            true_positives = confusion_matrix(y_true, y_pred)[int(class_label), int(class_label)]
            class_samples = list(y_true).count(int(class_label))
            accuracy_per_class[class_label] = true_positives / class_samples if class_samples > 0 else 0

    print(f"Accuracy per class: {accuracy_per_class}")

    # 保存每个类别的详细指标到文件
    if save_results:
        # 创建保存路径
        seed_folder_path = os.path.join(root_result_path, split_seed)
        if not os.path.exists(seed_folder_path):
            os.makedirs(seed_folder_path)
        datasettype_seed_folder_path = os.path.join(seed_folder_path, dataset_type)
        if not os.path.exists(datasettype_seed_folder_path):
            os.makedirs(datasettype_seed_folder_path)

        # 保存分类报告到CSV文件
        report_filename = os.path.join(datasettype_seed_folder_path, 'classification_report.csv')
        with open(report_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy'])  # 新增 Accuracy 列
            for class_label, metrics in class_report.items():
                if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
                    writer.writerow(
                        [class_label] + [metrics.get(x) for x in ['precision', 'recall', 'f1-score', 'support']] + [
                            accuracy_per_class.get(class_label, 0)])

        # 保存微平均和宏平均（AUC）等
        save_micro_average_roc_curve(y_true, all_scores, ['0', '1', '2', '3', '4', '5'], datasettype_seed_folder_path)
        save_micro_average_precision_recall_curve(y_true, all_scores, ['0', '1', '2', '3', '4', '5'],
                                                  datasettype_seed_folder_path)
        save_classification_metrics(y_true, y_pred, average_type, dataset_type, split_seed,
                                    datasettype_seed_folder_path)
        print('save all result done')


def extract_global_features0(dataset):
    """ Extract the 'global0' feature from each sample in the dataset. """
    return torch.stack([x['global0'] for x in dataset])

def extract_global_features(dataset):
    """ Extract the 'global0' feature from each sample in the dataset. """
    return torch.stack([x[0]['global0'] for x in dataset])
def replace_global_features(dataset, transformed_features):
    """ Replace the 'global0' feature in the original dataset with the transformed features. """
    for (sample, _), new_feature in zip(dataset, transformed_features):
        sample['global0'] = new_feature
    return dataset


def replace_global_features0(dataset, transformed_features):
    """ Replace the 'global0' feature in the original dataset with the transformed features. """
    for sample, new_feature in zip(dataset, transformed_features):
        sample['global0'] = new_feature
    return dataset


def transform_features(train_dataset, val_dataset, test_dataset):
    # Convert PyTorch datasets to lists for easier manipulation
    train_list = [train_dataset[i] for i in range(len(train_dataset))]
    val_list = [val_dataset[i] for i in range(len(val_dataset))]
    test_list = [test_dataset[i]for i in range(len(test_dataset))]

    # Convert PyTorch datasets to lists for easier manipulation
    # train_list = [train_dataset[i][0] for i in range(len(train_dataset))]  # Unpack the data part
    # val_list = [val_dataset[i][0] for i in range(len(val_dataset))]
    # test_list = [test_dataset[i][0] for i in range(len(test_dataset))]

    # Extract global features from each dataset
    X_train = extract_global_features0(train_list)
    X_val = extract_global_features0(val_list)
    X_test = extract_global_features0(test_list)

    # Convert tensors to numpy for scaling
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    X_test_np = X_test.numpy()

    # Initialize scalers
    scaler_min_max = MinMaxScaler()
    scaler_standard = StandardScaler()

    # Fit and transform training data
    X_train_np = scaler_min_max.fit_transform(X_train_np)
    X_train_np = scaler_standard.fit_transform(X_train_np)

    # Transform validation and test data using the fitted scalers
    X_val_np = scaler_min_max.transform(X_val_np)
    X_val_np = scaler_standard.transform(X_val_np)
    X_test_np = scaler_min_max.transform(X_test_np)
    X_test_np = scaler_standard.transform(X_test_np)

    # Convert numpy arrays back to tensors
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)

    # Replace the original global0 features with transformed features
    train_dataset = replace_global_features0(train_list, X_train_torch)
    val_dataset = replace_global_features0(val_list, X_val_torch)
    test_dataset = replace_global_features0(test_list, X_test_torch)

    return train_dataset, val_dataset, test_dataset


def extract_node_features(dataset):
    """ Extract the node features (x) from each graph in the dataset. """
    return torch.cat([data['gnn0'].x for data in dataset], dim=0)


def replace_node_features(dataset, transformed_features):
    """ Replace the node features (x) in the original graphs with the transformed features. """
    start_idx = 0
    for data in dataset:
        end_idx = start_idx + data['gnn0'].x.size(0)
        data['gnn0'].x = transformed_features[start_idx:end_idx]
        start_idx = end_idx
    return dataset


def extract_node_features1(dataset):
    """ Extract the node features (x) from each graph in the dataset. """
    return torch.cat([data['gnn1'].x for data in dataset], dim=0)


def replace_node_features1(dataset, transformed_features):
    """ Replace the node features (x) in the original graphs with the transformed features. """
    start_idx = 0
    for data in dataset:
        end_idx = start_idx + data['gnn1'].x.size(0)
        data['gnn1'].x = transformed_features[start_idx:end_idx]
        start_idx = end_idx
    return dataset

def extract_node_features2(dataset):
    """ Extract the node features (x) from each graph in the dataset. """
    return torch.cat([data['gnn2'].x for data in dataset], dim=0)


def replace_node_features2(dataset, transformed_features):
    """ Replace the node features (x) in the original graphs with the transformed features. """
    start_idx = 0
    for data in dataset:
        end_idx = start_idx + data['gnn2'].x.size(0)
        data['gnn2'].x = transformed_features[start_idx:end_idx]
        start_idx = end_idx
    return dataset

def transform_node_features(train_dataset, val_dataset, test_dataset):
    # Convert PyTorch datasets to lists for easier manipulation
    train_list = [train_dataset[i] for i in range(len(train_dataset))]
    val_list = [val_dataset[i] for i in range(len(val_dataset))]
    test_list = [test_dataset[i] for i in range(len(test_dataset))]

    # Convert PyTorch datasets to lists for easier manipulation
    # train_list = [train_dataset[i][0] for i in range(len(train_dataset))]  # Unpack the data part
    # val_list = [val_dataset[i][0] for i in range(len(val_dataset))]
    # test_list = [test_dataset[i][0] for i in range(len(test_dataset))]

    # Convert PyTorch datasets to lists for easier manipulation
    # train_list = [train_dataset[i][0] for i in range(len(train_dataset))]  # Unpack the data part
    # val_list = [val_dataset[i][0] for i in range(len(val_dataset))]
    # test_list = [test_dataset[i][0] for i in range(len(test_dataset))]

    # Extract node features from each dataset
    X_train = extract_node_features(train_list)
    X_val = extract_node_features(val_list)
    X_test = extract_node_features(test_list)

    # Convert tensors to numpy for scaling
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    X_test_np = X_test.numpy()

    # Initialize scalers
    scaler_min_max = MinMaxScaler()
    scaler_standard = StandardScaler()

    # Fit and transform training data
    X_train_np = scaler_min_max.fit_transform(X_train_np)
    X_train_np = scaler_standard.fit_transform(X_train_np)

    # Transform validation and test data using the fitted scalers
    X_val_np = scaler_min_max.transform(X_val_np)
    X_val_np = scaler_standard.transform(X_val_np)
    X_test_np = scaler_min_max.transform(X_test_np)
    X_test_np = scaler_standard.transform(X_test_np)

    # Convert numpy arrays back to tensors
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)

    # Replace the original node features with transformed features
    train_dataset = replace_node_features(train_list, X_train_torch)
    val_dataset = replace_node_features(val_list, X_val_torch)
    test_dataset = replace_node_features(test_list, X_test_torch)

    return train_dataset, val_dataset, test_dataset


def transform_node_features1(train_dataset, val_dataset, test_dataset):
    # Convert PyTorch datasets to lists for easier manipulation
    train_list = [train_dataset[i] for i in range(len(train_dataset))]
    val_list = [val_dataset[i] for i in range(len(val_dataset))]
    test_list = [test_dataset[i] for i in range(len(test_dataset))]

    # Convert PyTorch datasets to lists for easier manipulation
    # train_list = [train_dataset[i][0] for i in range(len(train_dataset))]  # Unpack the data part
    # val_list = [val_dataset[i][0] for i in range(len(val_dataset))]
    # test_list = [test_dataset[i][0] for i in range(len(test_dataset))]
    # Extract node features from each dataset
    X_train = extract_node_features1(train_list)
    X_val = extract_node_features1(val_list)
    X_test = extract_node_features1(test_list)

    # Convert tensors to numpy for scaling
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    X_test_np = X_test.numpy()

    # Initialize scalers
    scaler_min_max = MinMaxScaler()
    scaler_standard = StandardScaler()

    # Fit and transform training data
    X_train_np = scaler_min_max.fit_transform(X_train_np)
    X_train_np = scaler_standard.fit_transform(X_train_np)

    # Transform validation and test data using the fitted scalers
    X_val_np = scaler_min_max.transform(X_val_np)
    X_val_np = scaler_standard.transform(X_val_np)
    X_test_np = scaler_min_max.transform(X_test_np)
    X_test_np = scaler_standard.transform(X_test_np)

    # Convert numpy arrays back to tensors
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)

    # Replace the original node features with transformed features
    train_dataset = replace_node_features1(train_list, X_train_torch)
    val_dataset = replace_node_features1(val_list, X_val_torch)
    test_dataset = replace_node_features1(test_list, X_test_torch)

    return train_dataset, val_dataset, test_dataset

def transform_node_features2(train_dataset, val_dataset, test_dataset):
    # Convert PyTorch datasets to lists for easier manipulation
    train_list = [train_dataset[i] for i in range(len(train_dataset))]
    val_list = [val_dataset[i] for i in range(len(val_dataset))]
    test_list = [test_dataset[i] for i in range(len(test_dataset))]

    # Convert PyTorch datasets to lists for easier manipulation
    # train_list = [train_dataset[i][0] for i in range(len(train_dataset))]  # Unpack the data part
    # val_list = [val_dataset[i][0] for i in range(len(val_dataset))]
    # test_list = [test_dataset[i][0] for i in range(len(test_dataset))]
    # Extract node features from each dataset
    X_train = extract_node_features2(train_list)
    X_val = extract_node_features2(val_list)
    X_test = extract_node_features2(test_list)

    # Convert tensors to numpy for scaling
    X_train_np = X_train.numpy()
    X_val_np = X_val.numpy()
    X_test_np = X_test.numpy()

    # Initialize scalers
    scaler_min_max = MinMaxScaler()
    scaler_standard = StandardScaler()

    # Fit and transform training data
    X_train_np = scaler_min_max.fit_transform(X_train_np)
    X_train_np = scaler_standard.fit_transform(X_train_np)

    # Transform validation and test data using the fitted scalers
    X_val_np = scaler_min_max.transform(X_val_np)
    X_val_np = scaler_standard.transform(X_val_np)
    X_test_np = scaler_min_max.transform(X_test_np)
    X_test_np = scaler_standard.transform(X_test_np)

    # Convert numpy arrays back to tensors
    X_train_torch = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_torch = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_torch = torch.tensor(X_test_np, dtype=torch.float32)

    # Replace the original node features with transformed features
    train_dataset = replace_node_features2(train_list, X_train_torch)
    val_dataset = replace_node_features2(val_list, X_val_torch)
    test_dataset = replace_node_features2(test_list, X_test_torch)

    return train_dataset, val_dataset, test_dataset
def load_splits_from_h5py(h5_file_path, key):
    """
    Load and deserialize dataset splits from an h5py file.

    Parameters:
    - h5_file_path: Path to the h5py file.
    - key: The key under which the splits are stored.

    Returns:
    - splits: A list of dictionaries with keys 'train', 'validation', and 'test', each containing the corresponding indices.
    """
    with h5py.File(h5_file_path, 'r+') as h5_file:
        if key in h5_file:
            serialized_splits = h5_file[key][()]
            splits = pickle.loads(serialized_splits.tostring())
            return splits
        else:
            print(f"No splits found under the key '{key}'.")
            return None


def generate_datasets_for_split0(dataset, split):
    """
    Generate train, validation, and test datasets based on a single split.

    Parameters:
    - dataset: The original full dataset instance.
    - split: A dictionary with keys 'train', 'validation', and 'test' containing indices.

    Returns:
    - A tuple containing train, validation, and test dataset instances.
    """
    train_indices = split['train']
    val_indices = split['validation']
    test_indices = split['test']

    # Assuming CombinedDataset can be indexed by a list of indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    # print(test_dataset[0])
    # print(test_dataset[0]['gnn0'].x)
    # print(type(test_dataset[0]))
    # print(len(test_dataset[0]))
    # exit()
    return train_dataset, val_dataset, test_dataset


def generate_datasets_for_split1(dataset, split):
    train_indices = split['train']
    val_indices = split['validation']
    test_indices = split['test']

    train_dataset = IndexedDataset(torch.utils.data.Subset(dataset, train_indices))
    val_dataset = IndexedDataset(torch.utils.data.Subset(dataset, val_indices))
    test_dataset = IndexedDataset(torch.utils.data.Subset(dataset, test_indices))

    return train_dataset, val_dataset, test_dataset


def generate_datasets_for_split(dataset, split):
    train_indices = split['train']
    val_indices = split['validation']
    test_indices = split['test']

    # Assuming CombinedDataset can be indexed by a list of indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return IndexedDataset(train_dataset, train_indices), IndexedDataset(val_dataset, val_indices), IndexedDataset(
        test_dataset, test_indices)


# file_path = "C:/Users/11156\Desktop\gnn-project-main/street-network-data-0/image/selected-global-features/selected-gloabl-features.h5"
# dataset = CombinedDataset(file_path=file_path, config=config)
# print('construct dataset done')
# print(len(dataset))
# data_item = dataset[24]
# # print(data_item)
# # print(data_item['cnn'])
# # Now verify the content of this sample
# verify_dataset_content(data_item)
# exit()
# # Example: Fetch the first item
# data_item = dataset[0]

# Assuming you have a DataLoader
from torch.utils.data import DataLoader


# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
#
# for data in dataloader:
#     # Here, data is a batch where:
#     # data['cnn'] contains the batched images,
#     # data['gnn'] contains the batched graph data,
#     # data['global'] contains the batched global features
#     pass

# Define abstract base classes for each type of model component
class BaseCNN(nn.Module):
    def __init__(self):
        super(BaseCNN, self).__init__()


class BaseGNN(nn.Module):
    def __init__(self):
        super(BaseGNN, self).__init__()


class BaseGlobalModel(nn.Module):
    def __init__(self):
        super(BaseGlobalModel, self).__init__()


from torchvision import models


class ModifiedResNet34(nn.Module):
    def __init__(self, num_ftrs):
        super(ModifiedResNet34, self).__init__()
        self.resnet34 = models.resnet34(pretrained=True)
        self.resnet34.fc = nn.Identity()  # Use Identity to remove the final fully connected layer
        self.adjust_features = nn.Linear(512, num_ftrs)  # Adjust features to the desired size

    def forward(self, x):
        features = self.resnet34(x)
        adjusted_features = self.adjust_features(features)
        return adjusted_features


class CustomGNN(nn.Module):
    def __init__(self, input_feature_size, model_params):
        super(CustomGNN, self).__init__()
        # Initialize model parameters and layers as provided
        self.initialize_layers(input_feature_size, model_params)

    def initialize_layers(self, input_feature_size, model_params):
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = nn.ModuleList([])
        self.transf_layers = nn.ModuleList([])
        self.pooling_layers = nn.ModuleList([])
        self.bn_layers = nn.ModuleList([])

        self.conv1 = TransformerConv(input_feature_size,
                                     embedding_size,
                                     heads=n_heads,
                                     dropout=dropout_rate,
                                     edge_dim=edge_dim,
                                     beta=True)
        self.conv2 = ChebConv(input_feature_size,
                              embedding_size,
                              3
                              )
        self.transf1 = nn.Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.n_layers):
            self.conv_layers.append(ChebConv(embedding_size, embedding_size, 3))
            self.transf_layers.append(nn.Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.linear1 = nn.Linear(embedding_size * 2, dense_neurons)
        self.linear2 = nn.Linear(dense_neurons, int(dense_neurons / 2))
        self.linear3 = nn.Linear(int(dense_neurons / 2), 6)

    def forward(self, x, edge_attr, edge_index, batch_index):
        # x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)
        global_representation = []
        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)
            if i % self.top_k_every_n == 0 or i == self.n_layers - 1:  # Ensure it works for the last layer
                x, edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i / self.top_k_every_n)](
                    x, edge_index, edge_attr, batch_index
                )
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        x = sum(global_representation)
        # print(x.shape)
        # exit()
        return x


class EnhancedMLP(nn.Module):
    def __init__(self, input_features, output_features=256):
        super(EnhancedMLP, self).__init__()
        self.layer1 = nn.Linear(input_features, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(512, output_features)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(output_features, output_features)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.output_layer(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(FusionLayer, self).__init__()
        # Calculate total combined feature size
        self.total_feature_size = sum(input_dims)
        # Layers to process combined features
        self.fusion = nn.Sequential(
            nn.Linear(self.total_feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, inputs):
        # Concatenate all input features
        combined_features = torch.cat(inputs, dim=1)

        # Store the combined features for analysis
        global combined_features_storage  # Declare as global if used outside the class
        # combined_features_storage = combined_features.detach().cpu().numpy()
        # print("Appending features with shape:", combined_features.shape)

        combined_features_storage.append(combined_features.detach().cpu().numpy())

        # Map combined features to class predictions
        output = self.fusion(combined_features)

        # _, preds = output.max(1)
        # # print(preds)
        # # print(output)
        # # all_preds.extend(preds.cpu().numpy())
        # pred = []
        # pred.append(preds.cpu().numpy())
        # print(pred)
        # print(output.detach().cpu().numpy())
        # combined_features_storage.append(pred)
        return output
        # Map combined features to class predictions
        # return self.fusion(combined_features)


class DynamicModel(nn.Module):

    def __init__(self, config, num_classes):
        super(DynamicModel, self).__init__()
        self.models = nn.ModuleDict()
        self.feature_sizes = []

        if 'cnn' in config:
            self.models['cnn'] = ModifiedResNet34(num_ftrs=config['cnn']['features'])
            self.feature_sizes.append(config['cnn']['features'])

        if 'gnn0' in config:
            # print(gnn_key)
            self.models['gnn0'] = CustomGNN(input_feature_size=config['gnn0']['input_feature_size'],
                                            model_params=config['gnn0']['params'])
            self.feature_sizes.append(config['gnn0']['output_feature_size'])

        if 'gnn1' in config:
            # print(gnn_key)
            self.models['gnn1'] = CustomGNN(input_feature_size=config['gnn1']['input_feature_size'],
                                            model_params=config['gnn1']['params'])
            self.feature_sizes.append(config['gnn1']['output_feature_size'])

        if 'gnn2' in config:
            # print(gnn_key)
            self.models['gnn2'] = CustomGNN(input_feature_size=config['gnn2']['input_feature_size'],
                                            model_params=config['gnn2']['params'])
            self.feature_sizes.append(config['gnn2']['output_feature_size'])

        if 'global' in config:
            self.models['global'] = EnhancedMLP(input_features=config['global']['input_features'],
                                                output_features=config['global']['features'])
            self.feature_sizes.append(config['global']['features'])

        if 'global0' in config:
            self.models['global0'] = EnhancedMLP(input_features=config['global0']['input_features'],
                                                 output_features=config['global0']['features'])
            self.feature_sizes.append(config['global0']['features'])

        # print(self.feature_sizes)
        self.fusion_layer = FusionLayer(self.feature_sizes, num_classes)

    def get_cnn_parameters(self):
        # Retrieve parameters from all CNN components
        for name, module in self.models.items():
            if 'cnn' in name:
                for param in module.parameters():
                    yield param

    def get_gnn0_parameters(self):
        # Retrieve parameters from all GNN components
        for name, module in self.models.items():
            if 'gnn0' in name:
                for param in module.parameters():
                    yield param

    def get_gnn1_parameters(self):
        # Retrieve parameters from all GNN components
        for name, module in self.models.items():
            if 'gnn1' in name:
                for param in module.parameters():
                    yield param

    def get_gnn2_parameters(self):
        # Retrieve parameters from all GNN components
        for name, module in self.models.items():
            if 'gnn2' in name:
                for param in module.parameters():
                    yield param

    def forward(self, inputs):
        # print(inputs)
        outputs = []
        for key, model in self.models.items():
            base_key = key.split('_')[0]
            for input_key in inputs:
                # print(input_key)
                if base_key in input_key:  # Match base key with input keys
                    input_data = inputs[input_key]
                    if 'gnn0' in key:
                        # print(key)
                        # For GNN, unpack the required fields
                        output = model(x=input_data.x, edge_attr=input_data.edge_attr, edge_index=input_data.edge_index,
                                       batch_index=input_data.batch)
                    elif 'gnn1' in key:
                        # print(key)
                        # For GNN, unpack the required fields
                        output = model(x=input_data.x, edge_attr=input_data.edge_attr, edge_index=input_data.edge_index,
                                       batch_index=input_data.batch)

                    elif 'gnn2' in key:
                        # print(key)
                        # For GNN, unpack the required fields
                        output = model(x=input_data.x, edge_attr=input_data.edge_attr, edge_index=input_data.edge_index,
                                       batch_index=input_data.batch)
                    else:
                        # For CNN and MLP, pass the data directly
                        output = model(input_data)
                    outputs.append(output)
                    # print(f"Matched {key} with {input_key}")
                    break

        # return output
        return self.fusion_layer(outputs)


import os
import time
import csv
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import numpy as np


# 假设其他必要的import语句已经在文件顶部存在
def save_cost(training_time, model_parameters, split_seed, root_result_path):
    # 创建存储路径
    seed_folder_path = os.path.join(root_result_path, split_seed)
    if not os.path.exists(seed_folder_path):
        os.makedirs(seed_folder_path)

    # 文件路径
    cost_file_path = os.path.join(seed_folder_path, 'cost.csv')

    # 保存成本信息到CSV
    with open(cost_file_path, 'w', newline='') as cost_csv:
        csv_writer = csv.writer(cost_csv)
        csv_writer.writerow(['Metric', 'Value'])
        csv_writer.writerow(['Training Time (seconds)', training_time])
        csv_writer.writerow(['Number of Parameters', model_parameters])
    print(f'Cost saved to {cost_file_path}')

def train(model, train_loader, optimizers, criterion, device):
    global combined_features_storage
    combined_features_storage = []  # Clear previous features

    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_indices = []  # To store indices

    for data, indices in train_loader:

        # print(indices)
        # exit()

        # Move data to device and prepare inputs
        for key in data.keys():
            # print(key)
            # exit()
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)

        # Clear gradients for all optimizers
        for opt in optimizers.values():
            opt.zero_grad()

        if 'cnn' in data:
            data['cnn'] = data['cnn'].to(device)
        if 'gnn0' in data:
            data['gnn0'].x = data['gnn0'].x.to(device)
            data['gnn0'].edge_index = data['gnn0'].edge_index.to(device)
            data['gnn0'].edge_attr = data['gnn0'].edge_attr.to(device)
            data['gnn0'].batch = data['gnn0'].batch.to(device) if hasattr(data['gnn0'], 'batch') else None
        if 'gnn1' in data:
            data['gnn1'].x = data['gnn1'].x.to(device)
            data['gnn1'].edge_index = data['gnn1'].edge_index.to(device)
            data['gnn1'].edge_attr = data['gnn1'].edge_attr.to(device)
            data['gnn1'].batch = data['gnn1'].batch.to(device) if hasattr(data['gnn1'], 'batch') else None
        if 'gnn2' in data:
            data['gnn2'].x = data['gnn2'].x.to(device)
            data['gnn2'].edge_index = data['gnn2'].edge_index.to(device)
            data['gnn2'].edge_attr = data['gnn2'].edge_attr.to(device)
            data['gnn2'].batch = data['gnn2'].batch.to(device) if hasattr(data['gnn2'], 'batch') else None
        if 'global' in data:
            data['global'] = data['global'].to(device)
        if 'global0' in data:
            data['global0'] = data['global0'].to(device)

            # print(data['global0'])
            # print(type(data['global0']))
            # print(data['global0'].shape)

            # exit()

        # Compute the model output
        output = model(data)
        loss = criterion(output, data['label'].view(-1))  # Assuming 'label' is always present
        total_loss += loss.item()

        # Perform backpropagation
        loss.backward()

        # Step each optimizer
        for opt in optimizers.values():
            opt.step()

        # Optionally accumulate predictions and labels for further analysis
        _, preds = output.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data['label'].cpu().numpy())
        all_indices.extend(indices)
        # Store the combined features along with their indices

        # combined_features_storage.extend(output.detach().cpu().numpy())

        # combined_features_storage.extend(
        #     [(feature, idx) for feature, idx in zip(output.detach().cpu().numpy(), indices)])

    calculate_metrics(np.array(all_preds), np.array(all_labels))

    return total_loss / len(train_loader), all_indices


def evaluate(model, loader, criterion, device):
    global combined_features_storage
    combined_features_storage = []  # Clear previous features

    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_indices = []  # To store indices

    with torch.no_grad():
        for data, indices in loader:
            # Move data to device
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            if 'cnn' in data:
                data['cnn'] = data['cnn'].to(device)
            if 'gnn0' in data:
                data['gnn0'].x = data['gnn0'].x.to(device)
                data['gnn0'].edge_index = data['gnn0'].edge_index.to(device)
                data['gnn0'].edge_attr = data['gnn0'].edge_attr.to(device)
                data['gnn0'].batch = data['gnn0'].batch.to(device) if hasattr(data['gnn0'], 'batch') else None
            if 'gnn1' in data:
                data['gnn1'].x = data['gnn1'].x.to(device)
                data['gnn1'].edge_index = data['gnn1'].edge_index.to(device)
                data['gnn1'].edge_attr = data['gnn1'].edge_attr.to(device)
                data['gnn1'].batch = data['gnn1'].batch.to(device) if hasattr(data['gnn1'], 'batch') else None
            if 'gnn2' in data:
                data['gnn2'].x = data['gnn2'].x.to(device)
                data['gnn2'].edge_index = data['gnn2'].edge_index.to(device)
                data['gnn2'].edge_attr = data['gnn2'].edge_attr.to(device)
                data['gnn2'].batch = data['gnn2'].batch.to(device) if hasattr(data['gnn2'], 'batch') else None
            if 'global' in data:
                data['global'] = data['global'].to(device)
            if 'global0' in data:
                data['global0'] = data['global0'].to(device)

            # Compute the model output
            output = model(data)
            loss = criterion(output, data['label'].view(-1))  # Assuming 'label' is always present
            total_loss += loss.item()

            # Optionally accumulate predictions and labels for further analysis
            _, preds = output.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data['label'].cpu().numpy())
            all_indices.extend(indices)

            # combined_features_storage.extend(output.detach().cpu().numpy())
            # Store the combined features along with their indices
            # combined_features_storage.extend(
            #     [(feature, idx) for feature, idx in zip(output.detach().cpu().numpy(), indices)])

    average_loss = total_loss / len(loader)
    calculate_metrics(np.array(all_preds), np.array(all_labels))

    # calculate_metrics(np.array(all_preds), np.array(all_labels), np.array(all_scores), save_results, split_seed,
    #                   dataset_type, root_result_path)
    # Optionally, return predictions and labels for accuracy or other metric calculations
    # return average_loss, all_preds, all_labels
    # return average_loss, all_preds, all_labels
    return average_loss, all_indices


def evaluate1(model, loader, criterion, device,save_results=False,split_seed = None,dataset_type = None,root_result_path=None):
    global combined_features_storage
    combined_features_storage = []  # Clear previous features

    model.eval()
    total_loss = 0
    all_preds = []
    all_scores = []
    all_labels = []
    all_indices = []  # To store indices

    with torch.no_grad():
        for data, indices in loader:
            # Move data to device
            for key in data.keys():
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].to(device)

            if 'cnn' in data:
                data['cnn'] = data['cnn'].to(device)
            if 'gnn0' in data:
                data['gnn0'].x = data['gnn0'].x.to(device)
                data['gnn0'].edge_index = data['gnn0'].edge_index.to(device)
                data['gnn0'].edge_attr = data['gnn0'].edge_attr.to(device)
                data['gnn0'].batch = data['gnn0'].batch.to(device) if hasattr(data['gnn0'], 'batch') else None
            if 'gnn1' in data:
                data['gnn1'].x = data['gnn1'].x.to(device)
                data['gnn1'].edge_index = data['gnn1'].edge_index.to(device)
                data['gnn1'].edge_attr = data['gnn1'].edge_attr.to(device)
                data['gnn1'].batch = data['gnn1'].batch.to(device) if hasattr(data['gnn1'], 'batch') else None
            if 'gnn2' in data:
                data['gnn2'].x = data['gnn2'].x.to(device)
                data['gnn2'].edge_index = data['gnn2'].edge_index.to(device)
                data['gnn2'].edge_attr = data['gnn2'].edge_attr.to(device)
                data['gnn2'].batch = data['gnn2'].batch.to(device) if hasattr(data['gnn2'], 'batch') else None
            if 'global' in data:
                data['global'] = data['global'].to(device)
            if 'global0' in data:
                data['global0'] = data['global0'].to(device)

            # Compute the model output
            output = model(data)
            loss = criterion(output, data['label'].view(-1))  # Assuming 'label' is always present
            total_loss += loss.item()

            # Optionally accumulate predictions and labels for further analysis
            _, preds = output.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data['label'].cpu().numpy())
            pred1 = torch.softmax(output, 1)
            all_scores.extend(pred1.cpu().numpy())
            all_indices.extend(indices)

            # combined_features_storage.extend(output.detach().cpu().numpy())
            # Store the combined features along with their indices
            # combined_features_storage.extend(
            #     [(feature, idx) for feature, idx in zip(output.detach().cpu().numpy(), indices)])

    average_loss = total_loss / len(loader)
    # calculate_metrics(np.array(all_preds), np.array(all_labels))

    calculate_metrics(np.array(all_preds), np.array(all_labels), np.array(all_scores), save_results, split_seed,
                      dataset_type, root_result_path)
    # Optionally, return predictions and labels for accuracy or other metric calculations
    # return average_loss, all_preds, all_labels
    # return average_loss, all_preds, all_labels
    return average_loss


# 主函数
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 主函数
all_configs = []

# 定义所有可能的模型组合
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
    'gnn1': {
        'type': 'graph1',
        'columns': ['dual_graph_nx_list'],
        'input_feature_size': 4,
        'output_feature_size': 256,
        'params': {
            'model_embedding_size': 128,
            'model_attention_heads': 1,
            'model_layers': 2,
            'model_dropout_rate': 0.2,
            'model_top_k_ratio': 0.5,
            'model_top_k_every_n': 1,
            'model_dense_neurons': 256,
            'model_edge_dim': 1
        }
    },
    'gnn2': {
        'type': 'graph2',
        'columns': ['primal_graph_nx_list'],
        'input_feature_size': 5,
        'output_feature_size': 256,
        'params': {
            'model_embedding_size': 128,
            'model_attention_heads': 1,
            'model_layers': 2,
            'model_dropout_rate': 0.2,
            'model_top_k_ratio': 0.5,
            'model_top_k_every_n': 1,
            'model_dense_neurons': 256,
            'model_edge_dim': 1
        }
    },
    'global0': {'type': 'global', 'columns': ['global_handcrafted_features'], 'input_features': 23, 'features': 256},
    'label': {'type': 'label', 'columns': ['label0']}
}

# 生成所有合法组合
from itertools import combinations

# # 遍历单模型组合
# for key in ['cnn', 'gnn0', 'gnn1', 'gnn2', 'global0']:
#     all_configs.append({key: possible_models[key], 'label': possible_models['label']})
#
# # 遍历双模型组合（gnn0、gnn1、gnn2 互斥）
# for model1, model2 in combinations(['cnn', 'gnn0', 'gnn1', 'gnn2', 'global0'], 2):
#     if 'gnn' in model1 and 'gnn' in model2:
#         continue  # 跳过 gnns 互斥的情况
#     all_configs.append(
#         {model1: possible_models[model1], model2: possible_models[model2], 'label': possible_models['label']})
#
# # 遍历三模型组合
# for model1, model2, model3 in combinations(['cnn', 'gnn0', 'gnn1', 'gnn2', 'global0'], 3):
#     if sum('gnn' in model for model in [model1, model2, model3]) > 1:
#         continue  # 跳过 gnns 互斥的情况
#     all_configs.append({
#         model1: possible_models[model1],
#         model2: possible_models[model2],
#         model3: possible_models[model3],
#         'label': possible_models['label']
#     })

# 在所有组合中，把特定组合配置优先放入 `all_configs` 中
all_configs = []

# # 定义优先运行的组合模型
# preferred_model_combination = {
#     'cnn': possible_models['cnn'],
#     'gnn0': possible_models['gnn0'],
#     'global0': possible_models['global0'],
#     'label': possible_models['label']
# }

# 定义优先运行的组合模型
preferred_model_combination = {

    'gnn0': possible_models['gnn0'],

    'label': possible_models['label']
}
# 将特定组合加入到all_configs的开头，确保先跑cnn + gnn0 + global0
all_configs.append(preferred_model_combination)

# 遍历其他组合
for key in ['cnn', 'gnn0', 'gnn1', 'gnn2', 'global0']:
    all_configs.append({key: possible_models[key], 'label': possible_models['label']})

# 遍历双模型组合（gnn0、gnn1、gnn2 互斥）
for model1, model2 in combinations(['cnn', 'gnn0', 'gnn1', 'gnn2', 'global0'], 2):
    if 'gnn' in model1 and 'gnn' in model2:
        continue  # 跳过 gnns 互斥的情况
    all_configs.append(
        {model1: possible_models[model1], model2: possible_models[model2], 'label': possible_models['label']})

# 遍历三模型组合
for model1, model2, model3 in combinations(['cnn', 'gnn0', 'gnn1', 'gnn2', 'global0'], 3):
    if sum('gnn' in model for model in [model1, model2, model3]) > 1:
        continue  # 跳过 gnns 互斥的情况
    all_configs.append({
        model1: possible_models[model1],
        model2: possible_models[model2],
        model3: possible_models[model3],
        'label': possible_models['label']
    })

# 现在 `all_configs` 列表中，`cnn + gnn0 + global0` 组合会先执行


# 训练和测试循环
num_classes = 6
file_path = "C:/Users/11156/Desktop/gnn-project-main/street-network-data-0/image/selected-global-features/selected-gloabl-features.h5"
# root_result_path_base = "C:/Users/11156/Desktop/gnn-project-main/result-model-performance-revisedrealround1/"
root_result_path_base = "C:/Users/11156/Desktop/gnn-project-main/feature importance-revisedrealround1/"

criterion = torch.nn.CrossEntropyLoss()
all_splits = load_splits_from_h5py(file_path, 'splits_train_ratio_0.6')
splits_to_use = all_splits[:5]  # 只使用前5个split



for config in all_configs:
    # 确定当前配置的路径
    model_combination_name = '+'.join([key for key in config if key != 'label'])
    root_result_path = os.path.join(root_result_path_base, model_combination_name)

    if not os.path.exists(root_result_path):
        os.makedirs(root_result_path)

    dataset = CombinedDataset(file_path, config)
    print(f'dataset creation done for combination: {model_combination_name}')

    # 定义每个split
    for i, split in enumerate(splits_to_use):
        print(f'starting split seed: {i} for combination: {model_combination_name}')

        # 在训练循环之前记录开始时间
        start_time = time.time()

        train_dataset, val_dataset, test_dataset = generate_datasets_for_split0(dataset, split)

        # 选择正确的特征转换方法
        if 'gnn0' in config:
            train_dataset, val_dataset, test_dataset = transform_node_features(train_dataset, val_dataset, test_dataset)
        elif 'gnn1' in config:
            train_dataset, val_dataset, test_dataset = transform_node_features1(train_dataset, val_dataset,
                                                                                test_dataset)
        elif 'gnn2' in config:
            train_dataset, val_dataset, test_dataset = transform_node_features2(train_dataset, val_dataset,
                                                                                test_dataset)

        if 'global0' in config:
            train_dataset, val_dataset, test_dataset = transform_features(train_dataset, val_dataset, test_dataset)

        train_loader = DataLoader(IndexedDataset(train_dataset, split['train']), batch_size=32, shuffle=True,
                                  collate_fn=collate_data)
        val_loader = DataLoader(IndexedDataset(val_dataset, split['validation']), batch_size=32, shuffle=False,
                                collate_fn=collate_data)
        test_loader = DataLoader(IndexedDataset(test_dataset, split['test']), batch_size=32, shuffle=False,
                                 collate_fn=collate_data)

        # 初始化动态模型
        combined_model = DynamicModel(config, num_classes).to(device)
        print('dynamic model construction done')

        optimizers = {}
        if 'cnn' in config:
            optimizers['cnn'] = torch.optim.SGD(combined_model.get_cnn_parameters(), lr=0.001, momentum=0.9)
        if 'gnn0' in config:
            optimizers['gnn0'] = torch.optim.SGD(combined_model.get_gnn0_parameters(), lr=0.01, momentum=0.9)
        if 'gnn1' in config:
            optimizers['gnn1'] = torch.optim.SGD(combined_model.get_gnn1_parameters(), lr=0.01, momentum=0.9)
        if 'gnn2' in config:
            optimizers['gnn2'] = torch.optim.SGD(combined_model.get_gnn2_parameters(), lr=0.01, momentum=0.9)
        if 'global0' in config:
            optimizers['global0'] = torch.optim.Adam(combined_model.models['global0'].parameters(), lr=0.001)

        best_val_loss = float('inf')
        early_stopping_patience = 2
        early_stopping_counter = 0
        best_model_path = 'best_model.pth'

        for epoch in range(1, 2):
            print(f"Epoch {epoch} for combination: {model_combination_name}")
            # 训练
            train_loss, train_indices = train(combined_model, train_loader, optimizers, criterion, device)

            # 验证
            val_loss, val_indices = evaluate(combined_model, val_loader, criterion, device)
            print(f"Validation Loss: {val_loss}")

            # 测试
            test_loss, test_indices = evaluate(combined_model, test_loader, criterion, device)
            print(f"Epoch {epoch}, Test Loss: {test_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(combined_model.state_dict(), best_model_path)
                print("Best model updated")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break

        # 加载最佳模型并进行最终评估
        combined_model.load_state_dict(torch.load(best_model_path))
        final_test_loss = evaluate1(combined_model, test_loader, criterion, device, save_results=True,
                                    split_seed=str(i),
                                    dataset_type='test', root_result_path=root_result_path)
        print(f"Final Test Loss: {final_test_loss}")

        final_test_loss = evaluate1(combined_model, val_loader, criterion, device, save_results=True, split_seed=str(i),
                                    dataset_type='val', root_result_path=root_result_path)
        final_test_loss = evaluate1(combined_model, train_loader, criterion, device, save_results=True,
                                    split_seed=str(i),
                                    dataset_type='train', root_result_path=root_result_path)

        # 保存成本信息
        save_cost(time.time() - start_time, sum(p.numel() for p in combined_model.parameters()), str(i),
                  root_result_path)

        print(f"best Test Loss: {test_loss}")
        print(f'finish split seed: {i} for combination: {model_combination_name}')

print('All combinations completed')
