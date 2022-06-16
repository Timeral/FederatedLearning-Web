import copy

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from time import sleep, time
import collections
# from utils import progress_bar
from pdb import set_trace


class NoneFedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NoneFedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


class FedAvgMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FedAvgMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.flatten(1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x


NumEpochs = 10
HiddenSize = 256
LearningRate = 0.015


# none federated MLP solution for MNIST
def non_fed_mlp_main(dataset_train, dataset_test):
    print("non_fed_mlp_main:")

    train_loader = DataLoader(
        dataset_train, batch_size=BatchSize, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        dataset_test, batch_size=len(dataset_test), num_workers=8)

    non_fed_net = NoneFedMLP(28 * 28, HiddenSize, 10).to(device)
    optimizer = optim.SGD(non_fed_net.parameters(), lr=LearningRate)

    epoch_loss, epoch_accuracy = None, None
    for epoch in range(NumEpochs):
        start_time = time()
        train_losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            out = non_fed_net(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        ans = None
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                out = torch.argmax(non_fed_net(data), dim=1)
                ans = (out == target).sum()
        epoch_loss, epoch_accuracy = sum(
            train_losses) / len(train_losses), ans / len(dataset_test)
        print(
            f"NonFed_MLP epoch {epoch}: loss={epoch_loss:.2f} accuracy={epoch_accuracy:.2f} ({time() - start_time:.2f}s)")
        if(epoch == NumEpochs-1):  # print parameters
            Total_params = 0
            Trainable_params = 0
            NonTrainable_params = 0
            for name, param in non_fed_net.named_parameters():
                print(name, param.size())
                if param.requires_grad:
                    Trainable_params += param.nelement()
                else:
                    NonTrainable_params += param.nelement()
            print("Trainable_params:", Trainable_params)
            print("NonTrainable_params:", NonTrainable_params)
    print("=" * 50)
    return epoch_loss, epoch_accuracy


NumClients = 3
BatchSize = 64


def split_dataset(dataset):
    num_items = len(dataset) // NumClients
    assert num_items > 0
    permutation = np.arange(len(dataset))
    np.random.shuffle(permutation)
    choice = [permutation[i * num_items:min(len(dataset), (i + 1) * num_items)]
              for i in range(NumClients)]
    split_data_loader = []
    for i in range(NumClients):
        cur_data = []
        cur_target = []
        for elem in choice[i]:
            cur_data.append(dataset[elem][0].numpy())
            cur_target.append(dataset[elem][1])
        cur_data = np.array(cur_data)
        cur_target = np.array(cur_target)
        split_data_loader.append(DataLoader(
            TensorDataset(torch.Tensor(cur_data),
                          torch.LongTensor(cur_target)),
            batch_size=BatchSize, shuffle=True, num_workers=8
        ))
    return split_data_loader


KClientsPerEpoch = NumClients
LocalEpoch = 3  # NumClients // KClientsPerEpoch

encrypt_para = [i for i in range(1, NumClients+1)]


def fedavg_mlp_main(dataset_train, dataset_test):  # federated MLP solution for MNIST
    print("fedavg_mlp_main:")
    fed_train_loader = split_dataset(dataset_train)
    test_loader = DataLoader(
        dataset_test, batch_size=len(dataset_test), num_workers=8)

    fed_net_global = FedAvgMLP(28 * 28, HiddenSize, 10).to(device)
    # optimizer = optim.SGD(fed_net_global.parameters(), lr=LearningRate)

    def client_update(train_loader, local_net, id):
        optimizer = optim.SGD(local_net.parameters(), lr=LearningRate)
        main_loss = []
        for _ in range(LocalEpoch):
            train_losses = []
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                out = local_net(data)
                loss = F.cross_entropy(out, target)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            main_loss.append(sum(train_losses) / len(train_losses))
        encrypt_weights = {}
        for k, v in local_net.state_dict().items():
            encrypt_weights[k] = copy.deepcopy(v) + encrypt_para[id]
        return encrypt_weights, sum(main_loss) / len(main_loss)
        # return local_net.state_dict(), sum(main_loss) / len(main_loss)

    epoch_loss, epoch_accuracy = None, None
    for epoch in range(NumEpochs):
        start_time = time()
        # participants = np.random.choice(
        #     np.arange(NumClients), KClientsPerEpoch, replace=False)
        participants = np.arange(NumClients)
        weights, losses = [], []
        for partipant_id in participants:
            net = copy.deepcopy(fed_net_global).to(device)
            train_loader = fed_train_loader[partipant_id]
            new_weight, avg_loss = client_update(
                train_loader, net, partipant_id)
            weights.append(copy.deepcopy(new_weight))
            losses.append(avg_loss)
        next_params = collections.OrderedDict()
        for k in fed_net_global.state_dict().keys():
            cur = None
            for elem in weights:
                if cur is None:
                    cur = copy.deepcopy(elem[k])
                else:
                    cur += elem[k]
            for i in encrypt_para:
                cur -= i
            cur = torch.div(cur, len(weights))
            next_params[k] = copy.deepcopy(cur)
        fed_net_global.load_state_dict(next_params)
        ans = None
        with torch.no_grad():
            for _, (data, target) in enumerate(test_loader):
                data = data.to(device)
                target = target.to(device)
                out = torch.argmax(fed_net_global(data), dim=1)
                ans = (out == target).sum()
        epoch_loss, epoch_accuracy = sum(
            losses) / len(losses), ans / len(dataset_test)
        print(
            f"FedAVG_MLP epoch {epoch}: loss={epoch_loss:.2f} accuracy={epoch_accuracy:.2f} ({time() - start_time:.2f}s)")
        # set_trace()
    print("=" * 50)
    return epoch_loss, epoch_accuracy


def main():
    # download dataset
    trans_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset_train = datasets.MNIST(
        '../data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST(
        '../data/mnist/', train=False, download=True, transform=trans_mnist)
    print(f"train: {len(dataset_train)}, test: {len(dataset_test)}")

    #non_fed_mlp_main(dataset_train, dataset_test)
    fedavg_mlp_main(dataset_train, dataset_test)


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available(), torch.version.cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"running on {device}")
    print("=" * 50)
    sleep(1)
    main()
