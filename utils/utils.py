import os
import torch
from torch.utils.data import DataLoader, TensorDataset

def mkdirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_dataloader(args, dataidxs=None):
    # Dummy data for demonstration
    data = torch.randn(100, 13)  # 100 samples, 13 features
    labels = torch.randint(0, 2, (100,))  # Binary classification

    if dataidxs is not None:
        data = data[dataidxs]
        labels = labels[dataidxs]

    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader, dataset

def partition_data(args):
    # Dummy partitioning
    net_dataidx_map = {0: list(range(100))}
    traindata_cls_counts = torch.zeros((1, 2))  # 1 client, 2 classes
    return net_dataidx_map, traindata_cls_counts

def partition_split_test_data(args):
    # Dummy partitioning
    split_test_dataidx = [list(range(100))]
    imbalanced_test_dist = [0.5, 0.5]  # Equal distribution for demonstration
    return split_test_dataidx, None, imbalanced_test_dist

def get_dataloader_split_test(args, split_test_dataidx):
    return get_dataloader(args, split_test_dataidx)

def init_nets(n_parties, args, device):
    input_dim = 13  # Assuming 13 features based on the provided data
    hidden_dims = [64, 32, 16]
    output_dim = 2  # Assuming binary classification
    nets = {i: SimpleMLP(input_dim, hidden_dims, output_dim).to(device) for i in range(n_parties)}
    return nets
