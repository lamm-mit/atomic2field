from os import WIFCONTINUED
import numpy as np
import os.path as osp
import time
import sklearn
import itertools
from sklearn.model_selection import train_test_split
import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data, DataLoader, DataListLoader
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, GRUCell
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool, global_sort_pool, DataParallel
from torch_geometric.utils import to_dense_batch


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.node_emb = Embedding(21, 75)
        self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=50, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, 1))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x.squeeze())
        print(x.size())
        edge_attr = self.edge_emb(edge_attr)

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))

        x = global_add_pool(x, batch)
        return self.mlp(x)


class PNANet(torch.nn.Module):
    def __init__(self):
        super(PNANet, self).__init__()
        
        
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.convs = ModuleList()
        self.batch_norms  = ModuleList()
        self.grus = ModuleList()
        
        for i in range(14):
            if i == 0:
                conv = PNAConv(in_channels=2, out_channels=50, aggregators=aggregators, scalers=scalers, deg=deg,
                          towers=1, pre_layers=1, post_layers=1, divide_input=False)
                self.convs.append(conv)
                self.grus.append(GRUCell(2, 50))
                self.batch_norms.append(BatchNorm(50))
            else:
                conv = PNAConv(in_channels=50, out_channels=50, aggregators=aggregators, scalers=scalers, deg=deg,
                          towers=5, pre_layers=1, post_layers=1, divide_input=False)
                self.convs.append(conv)
                self.grus.append(GRUCell(50, 50))
                self.batch_norms.append(BatchNorm(50))
                
        self.readout = PNAConv(in_channels=50, out_channels=1, aggregators=aggregators, scalers=scalers, deg=deg,
                          towers=1, pre_layers=1, post_layers=1, divide_input=False)

        
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        for conv, gru, batch_norm in zip(self.convs, self.grus, self.batch_norms):
            y = conv(x, edge_index)
            x = gru(x, y)
            x = F.relu(batch_norm(x))
        x = self.readout(x, edge_index)

        return x

class WeightLoss(nn.Module):
    def __init__(self, weight=None, size_average=None):
        super(WeightLoss, self).__init__()


    def forward(self, inputs, targets):        
        
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        loss_unweight = torch.square(inputs - targets)
        penalty = torch.heaviside(inputs - 0.7 * max_value, torch.tensor([0.0]).to(torch.float32).to(device))
        weight = 1 + 10 * penalty
        loss_weight = weight * loss_unweight 
        
        return torch.mean(loss_weight)


def train(model, dataloader, optimizer, device):
    batch_loss = []
    model.train()
    
    for batch in dataloader:
        #node, edge, label, idx = batch.x, batch.edge_index, batch.y, batch.batch
        #node = node.to(device)
        #edge = edge.to(device)
        #label = label.to(device)
        #idx = idx.to(device)
        
        # Train the model on each batch
        pred = model(batch)
        label = torch.cat([data.y for data in batch]).to(device)
        lst = range(len(batch))
        idx = torch.LongTensor(list(itertools.chain.from_iterable(itertools.repeat(x, 5928) for x in lst))).to(device)
        loss_mse = F.mse_loss(pred.squeeze(), label.squeeze())
        loss_ave = F.mse_loss(torch.mean(global_sort_pool(pred, idx, 300), 1).squeeze(), torch.mean(global_sort_pool(label.view(-1, 1), idx, 300), 1).squeeze())
        loss = loss_mse + 10 * loss_ave
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        
    return np.mean(np.array(batch_loss))

def validate(model, dataloader, device):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            #node, edge, label, idx = batch.x, batch.edge_index, batch.y, batch.batch
            #node = node.to(device)
            #edge = edge.to(device)
            #label = label.to(device)
            #idx = idx.to(device)
        
        # Train the model on each batch
            pred = model(batch)
            label = torch.cat([data.y for data in batch]).to(device)
            lst = range(len(batch))
            idx = torch.LongTensor(list(itertools.chain.from_iterable(itertools.repeat(x, 5928) for x in lst))).to(device)
            loss_mse = F.mse_loss(pred.squeeze(), label.squeeze())
            loss_ave = F.mse_loss(torch.mean(global_sort_pool(pred, idx, 300), 1).squeeze(), torch.mean(global_sort_pool(label.view(-1, 1), idx, 300), 1).squeeze())
            loss = loss_mse + 10 * loss_ave
            val_loss.append(loss.item())
    return np.mean(np.array(val_loss))


def write_data(pred_test, label_test, x_test):
    f1 = open("pred_sym.txt", "w")
    f2 = open("label_sym.txt", "w")
    f3 = open("x_sym.txt", "w")
    f4 = open("y_sym.txt", "w")
    num_data = len(pred_test)
    for i in range(num_data):
       num_graph = len(pred_test[i])
       for j in range(num_graph):
          f1.write(str(pred_test[i][j]) + "\t")
          f2.write(str(label_test[i][j]) + "\t")
          f3.write(str(x_test[i][j][0]) + "\t")
          f4.write(str(x_test[i][j][1]) + "\t")
       f1.write("\n")
       f2.write("\n")
       f3.write("\n")
       f4.write("\n")
    f1.close()
    f2.close()
    f3.close()
    f4.close()

if __name__ == "__main__":
    # Read relevant data files
    f1 = open("./edge_sym.txt", "r")
    f2 = open("./node_features_sym.txt", "r")
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    num_data = 6996
    data_list = []
    t0 = time.time()
    print("Number of data processed\ttime")
    ave = []
    for i in range(num_data):
        if i % 200 == 0:
            print(i, time.time() - t0)
        node1 = [int(idx) for idx in lines1[2 * i].split()[1:]]
        node2 = [int(idx) for idx in lines1[2 * i + 1].split()[1:]]
        edge_index = torch.tensor([node1, node2], dtype=torch.long)
        xs = [float(idx) for idx in lines2[2 * i].split()[1:]]
        ys = [float(idx) for idx in lines2[2 * i + 1].split()[1:]]
        node_feature = [[xs[j], ys[j]] for j in range(len(xs))]
        x = torch.tensor(node_feature, dtype=torch.float)
        node_label = [float(idx) * 0.0 for idx in range(len(xs))]
        for idx in range(len(xs)):
           ave.append(float(idx) * 0.0)
        y = torch.tensor(node_label, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    mean_value = np.mean(np.array(ave))
    Train_data, Test_data = train_test_split(data_list, test_size = 0.2, random_state=103)
    Train_data, Val_data = train_test_split(Train_data, test_size = 0.125, random_state=103)

    batch_size = 16
    train_loader = DataListLoader(Train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataListLoader(Test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataListLoader(Val_data, batch_size=batch_size, shuffle=True)
    all_loader = DataListLoader(data_list, batch_size=batch_size, shuffle=False)

    deg = torch.zeros(4, dtype=torch.long)
    for data in Train_data:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    device = "cuda"
    torch.cuda.empty_cache()
    model = PNANet()
    model = DataParallel(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, min_lr=1e-5, verbose=True)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model architecture:")
    print(model)
    print("The number of trainable parameters is:{}".format(params))

    # testing
    ckpt = torch.load('./ckpt/500.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    pred_test = []
    label_test = []
    x_test = []
    for batch in all_loader:
       #node, edge, label, batch = data.x, data.edge_index, data.y, data.batch
       #node = node.to(device)
       #edge = edge.to(device)
       #label = label.to(device)
        
       # Test the model on each batch
       with torch.no_grad():
         pred = model(batch)
       label = torch.cat([data.y for data in batch]).to(device)
       node = torch.cat([data.x for data in batch]).to(device)
       split_size = [5928 for i in range(len(batch))]
       pred_split = torch.split(pred, split_size)
       label_split = torch.split(label, split_size)
       x_split = torch.split(node, split_size, dim=0)
       num_graphs = len(pred_split)
       for i in range(num_graphs):
          pred_test.append(pred_split[i].cpu().detach().numpy().squeeze().tolist())
          label_test.append(label_split[i].cpu().detach().numpy().tolist())
          x_test.append(x_split[i].cpu().detach().numpy().tolist())
       torch.cuda.empty_cache()

    write_data(pred_test, label_test, x_test)

