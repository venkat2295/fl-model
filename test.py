import os
import json
import copy
import torch
import random
import logging
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from args import get_args
from disco import *
from algorithms import *
from utils.model import *
from utils.utils import *

# Define a simple neural network
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=2):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc4(x)
        return out

def get_dataloader(args, dataidxs=None):
    # Load data from CSV files
    train_data = pd.read_csv('./data/train.csv')
    test_data = pd.read_csv('./data/test.csv')

    # Assuming the last column is the label
    train_features = train_data.iloc[:, :-1].values
    train_labels = train_data.iloc[:, -1].values
    test_features = test_data.iloc[:, :-1].values
    test_labels = test_data.iloc[:, -1].values

    if dataidxs is not None:
        train_features = train_features[dataidxs]
        train_labels = train_labels[dataidxs]

    train_dataset = TensorDataset(torch.tensor(train_features, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def train_and_evaluate(args, device):
    # Load dataset
    train_dl, test_dl = get_dataloader(args)

    # Initialize model
    input_dim = 13  # Assuming 13 features based on the provided data
    hidden_dims = [64, 32, 16]
    output_dim = 2  # Assuming binary classification
    model = SimpleMLP(input_dim, hidden_dims, output_dim).to(device)

    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    num_epochs = 1000
    loss_values = []
    accuracy_values = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_dl)
        loss_values.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss}")

        # Evaluation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        accuracy_values.append(accuracy)
        print(f"Accuracy after Epoch {epoch + 1}: {accuracy}%")

    # Smoothing the loss and accuracy values
    window_size = 10
    smoothed_loss = moving_average(loss_values, window_size)
    smoothed_accuracy = moving_average(accuracy_values, window_size)

    # Plotting loss and accuracy
    epochs = range(1, num_epochs + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs[:len(smoothed_loss)], smoothed_loss, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(epochs[:len(smoothed_accuracy)], smoothed_accuracy, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title('Loss and Accuracy over Epochs')
    plt.show()

if __name__ == '__main__':
    # Configurations
    args = get_args()

    # Create necessary directories
    mkdirs(args.logdir)
    mkdirs(args.modeldir)
    mkdirs(args.datadir)

    now_time = datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    print(now_time)

    dataset_logdir = os.path.join(args.logdir, args.dataset)
    mkdirs(dataset_logdir)
    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s.json' % (now_time)
    else:
        argument_path = args.log_file_name + '.json'
    with open(os.path.join(dataset_logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (now_time)
    log_path = args.log_file_name + '.log'
    logging.basicConfig(
        filename=os.path.join(dataset_logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')
    logger = logging.getLogger()

    # Modified device setup
    use_cuda = torch.cuda.is_available() and args.device.startswith('cuda')
    device = torch.device('cuda' if use_cuda else 'cpu')

    if use_cuda:
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Using CPU")

    seed = args.init_seed
    logger.info("#" * 100)

    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    # Convert download_data to boolean for dataset loading
    args.download = bool(args.download_data)

    # Train and evaluate the model
    train_and_evaluate(args, device)

    print('>> Done!')
