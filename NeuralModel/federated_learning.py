from ast import Num
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

# from phe import paillier

import random
# import diffie_hellman
# from decimal import *
import shamir


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


NumEpochs = 5
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
    # for k, v in non_fed_net.state_dict().items():
    #     print(v[10])
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
precision_num = 2


def qsm(a, b, p):
    res = 1
    while(b):
        if b & 1:
            res = res*a % p
        a = a*a % p
        b >>= 1
    return res % p


def scale_num(num, k=precision_num):
    l = len(str(num))
    if l <= k:
        return num
    # num = num*(10**(k-l))
    num = str(num)
    res = num[0:k]
    res += '.'
    res += num[k:]
    return float(res)


mod = 998244353
diffie_hellman_g = 19260817
diffie_hellman_secretkey = []
diffie_hellman_publickey = []
diffie_hellman_commonkey = []


encrypt_para = [0 for _ in range(NumClients)]

shamir_share = [[] for _ in range(NumClients)]
shamir_threshold = NumClients >> 1 | 1
fix_drop_rate = 0.2
var_drop_rate = 0.1
n_drop_rate = 0
drop_rate = fix_drop_rate+var_drop_rate*n_drop_rate


def diffie_hellman_init():
    global diffie_hellman_secretkey
    global diffie_hellman_publickey
    diffie_hellman_secretkey = [random.randint(
        1e20, 9e20) for _ in range(NumClients)]
    diffie_hellman_publickey = [qsm(diffie_hellman_g, i, mod)
                                for i in diffie_hellman_secretkey]


def diffie_hellman_exchange():
    global diffie_hellman_commonkey
    for i in range(NumClients):
        diffie_hellman_commonkey.append(
            [scale_num(qsm(diffie_hellman_publickey[j], diffie_hellman_secretkey[i], mod), precision_num) for j in range(NumClients)])
    for i in range(NumClients):
        diffie_hellman_commonkey[i][i] = 0
    print("diffie_hellman_commonkey:", diffie_hellman_commonkey)


def fedavg_mlp_main(dataset_train, dataset_test):  # federated MLP solution for MNIST
    print("fedavg_mlp_main:")
    fed_train_loader = split_dataset(dataset_train)
    test_loader = DataLoader(
        dataset_test, batch_size=len(dataset_test), num_workers=8)

    fed_net_global = FedAvgMLP(28 * 28, HiddenSize, 10).to(device)
    # fed_para_global = []
    # optimizer = optim.SGD(fed_net_global.parameters(), lr=LearningRate)
    diffie_hellman_init()
    diffie_hellman_exchange()
    for i in range(NumClients):
        shares = shamir.generate_shares(
            NumClients, shamir_threshold, diffie_hellman_secretkey[i])
        for j in range(NumClients):
            shamir_share[j].append(shares[j])

    def client_update(train_loader, local_net, id):
        # local_net.load_state_dict(fed_para_global)
        optimizer = optim.SGD(local_net.parameters(), lr=LearningRate)
        main_loss = []
        global encrypt_para
        encrypt_para[id] = scale_num(random.randint(1e20, 9e20), precision_num)
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
            encrypt_weights[k] = copy.deepcopy(v)
            # print(type(encrypt_weights[k]))
            # print(encrypt_weights[k].shape)
            # print(encrypt_weights[k])
            # encrypt_weights[k] = encrypt_weights[k].cpu().numpy()
        for k, v in local_net.state_dict().items():
            for i in range(0, id):
                encrypt_weights[k] = encrypt_weights[k] - \
                    diffie_hellman_commonkey[id][i]
            for i in range(id+1, NumClients):
                encrypt_weights[k] = encrypt_weights[k] + \
                    diffie_hellman_commonkey[id][i]
            encrypt_weights[k] = encrypt_weights[k] + encrypt_para[id]
        return encrypt_weights, sum(main_loss) / len(main_loss)
        # return local_net.state_dict(), sum(main_loss) / len(main_loss)

    epoch_loss, epoch_accuracy = None, None
    for epoch in range(NumEpochs):
        start_time = time()
        participant_NumClients = NumClients
        global drop_rate, n_drop_rate, fix_drop_rate, var_drop_rate
        drop_rate = fix_drop_rate+var_drop_rate*n_drop_rate
        drop_judge = random.randrange(0, 10)*0.1 <= drop_rate
        if drop_judge:
            n_drop_rate = 0
            participant_NumClients -= 1
            print("drop_rate:", drop_rate)
        else:
            n_drop_rate += 1
        participants = np.random.choice(
            np.arange(NumClients), participant_NumClients, replace=False)
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
            # print("cur:", cur)
            # cur = cur.astype(float)
            # cur = torch.from_numpy(cur)
            for partipant_id in participants:
                cur = cur-encrypt_para[partipant_id]
            if drop_judge:
                drop_list = []
                for i in range(NumClients):
                    if i not in participants:
                        drop_list.append(i)
                reconstruction = []
                # print("drop_list:", drop_list)
                for i in drop_list:
                    for j in participants:
                        reconstruction.append(shamir_share[j][i])
                    pool = random.sample(reconstruction, shamir_threshold)
                    recon_secret = shamir.reconstruct_secret(pool)
                    # if recon_secret != diffie_hellman_secretkey[i]:
                    # print("key different")
                    # print("drop_client_id:", i)
                    # print("drop_client_secretkey:", recon_secret)
                    for j in range(0, i):
                        cur = cur - \
                            scale_num(
                                qsm(diffie_hellman_publickey[j], recon_secret, mod))
                    for j in range(i+1, NumClients):
                        cur = cur + \
                            scale_num(
                                qsm(diffie_hellman_publickey[j], recon_secret, mod))
            cur = torch.div(cur, len(weights))
            next_params[k] = copy.deepcopy(cur)
        fed_net_global.load_state_dict(next_params)
        # fed_para_global = next_params
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

    # non_fed_mlp_main(dataset_train, dataset_test)
    fedavg_mlp_main(dataset_train, dataset_test)


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available(), torch.version.cuda)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"running on {device}")
    print("=" * 50)
    sleep(1)
    main()
