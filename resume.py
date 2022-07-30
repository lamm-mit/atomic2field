from os import WIFCONTINUED
import numpy as np
import os.path as osp
import time
import sklearn
from sklearn.model_selection import train_test_split
import torch
import torch_geometric
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import degree
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear, GRUCell
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool



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

        
    def forward(self, x, edge_index):
        
        for conv, gru, batch_norm in zip(self.convs, self.grus, self.batch_norms):
            y = conv(x, edge_index)
            x = gru(x, y)
            x = F.relu(batch_norm(x))
        x = self.readout(x, edge_index)

        return x


def train(model, dataloader, optimizer, device):
    batch_loss = []
    model.train()
    
    for batch in dataloader:
        node, edge, label, idx = batch.x, batch.edge_index, batch.y, batch.batch
        node = node.to(device)
        edge = edge.to(device)
        label = label.to(device)
        idx = idx.to(device)
        
        # Train the model on each batch
        pred = model(node, edge)
        loss_mse = F.mse_loss(pred.squeeze(), label.squeeze())
        loss = loss_mse
        
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
            node, edge, label, idx = batch.x, batch.edge_index, batch.y, batch.batch
            node = node.to(device)
            edge = edge.to(device)
            label = label.to(device)
            idx = idx.to(device)
        
        # Train the model on each batch
            pred = model(node, edge)
            loss_mse = F.mse_loss(pred.squeeze(), label.squeeze())
            loss = loss_mse
            val_loss.append(loss.item())
    return np.mean(np.array(val_loss))


if __name__ == "__main__":
    # Read relevant data files
    f1 = open("//pool001/zhenzey/atomic2field/GrapheneVacancy/ML/GNN/edge_conc_0.1.txt", "r")
    f2 = open("//pool001/zhenzey/atomic2field/GrapheneVacancy/ML/GNN/node_features_conc_0.1.txt", "r")
    f3 = open("//pool001/zhenzey/atomic2field/GrapheneVacancy/ML/GNN/node_labels_sxx_conc_0.1.txt", "r")
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()
    num_data = 2000
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
        node_label = [float(idx) * 1e-6 for idx in lines3[i].split()[1:]]
        for idx in lines3[i].split()[1:]:
           ave.append(float(idx) * 1e-6)
        y = torch.tensor(node_label, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    mean_value = np.mean(np.array(ave))
    Train_data, Test_data = train_test_split(data_list, test_size = 0.2, random_state=42)
    Train_data, Val_data = train_test_split(Train_data, test_size = 0.125, random_state=42)

    batch_size = 8
    train_loader = DataLoader(Train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Val_data, batch_size=batch_size, shuffle=True)

    deg = torch.zeros(4, dtype=torch.long)
    for data in Train_data:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    device = "cuda:0"
    torch.cuda.empty_cache()
    model = PNANet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, min_lr=-1e-5, verbose=True)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model architecture:")
    print(model)
    print("The number of trainable parameters is:{}".format(params))
     
    # load checkpoint
    ckpt = torch.load('./ckpt/400.pt')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch']

    path = './ckpt/'
    # Training
    print("epoch", "train loss", "validation loss")

    val_loss_curve = []
    train_loss_curve = []

    for epoch in range(start_epoch, 500):
    
        # Compute train your model on training data
        epoch_loss = train(model, train_loader, optimizer,  device=0)
    
        # Validate your on validation data 
        val_loss = validate(model, val_loader, device=0)     

    
        # Record train and loss performance 
        train_loss_curve.append(epoch_loss)
        val_loss_curve.append(val_loss)
    
        # The learning rate scheduler record the validation loss 
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch, 
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':epoch_loss,
            
            },
            path + str(epoch+1) + ".pt")
        print(epoch, epoch_loss, val_loss)
