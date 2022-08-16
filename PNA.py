from os import WIFCONTINUED
import numpy as np
import os.path as osp
import time
import sklearn
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
from torch_geometric.nn import PNAConv, BatchNorm, global_mean_pool, DataParallel
import argparse

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', default=8, type=int,
                    help='batch size')
parser.add_argument('-d','--data_path', default='./data/', type=str,
                    help='data path')
parser.add_argument('-i','--input_dim', default=2, type=int,
                    help='the dimension of coordinates (2D or 3D)')
parser.add_argument('-n','--num_data', default=2000, type=int,
                    help='the number of all data')
parser.add_argument('-l','--num_layer', default=14, type=int,
                    help='the number of PNAConv layers')
parser.add_argument('-v','--hidden_dim', default=50, type=int,
                    help='the hidden dimension of PNANet')
parser.add_argument('-m','--max_degree', default=4, type=int,
                    help='maximum degree of all nodes')
parser.add_argument('-e','--epoch', default=500, type=int,
                    help='number of epoch')
parser.add_argument('-s','--scale_factor', default=1e-6, type=float,
                    help='scale factor for node labels')
args = parser.parse_args()

if args.hidden_dim % 5 != 0:
    raise Exception("Sorry, not available hidden dimension, need to be multiple of 5")
if args.num_layer < 1:
    raise Exception("Sorry, the number of layer is not enough")

# DL architecture
class PNANet(torch.nn.Module):
    def __init__(self):
        super(PNANet, self).__init__()
        
        
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        
        self.convs = ModuleList()
        self.batch_norms  = ModuleList()
        self.grus = ModuleList()
        
        num_layer = args.num_layer
        input_dim = args.input_dim
        hidden_dim = args.hidden_dim

        for i in range(num_layer):
            if i == 0:
                conv = PNAConv(in_channels=input_dim, out_channels=hidden_dim, aggregators=aggregators, scalers=scalers, deg=deg,
                          towers=1, pre_layers=1, post_layers=1, divide_input=False)
                self.convs.append(conv)
                self.grus.append(GRUCell(input_dim, hidden_dim))
                self.batch_norms.append(BatchNorm(hidden_dim))
            else:
                conv = PNAConv(in_channels=hidden_dim, out_channels=hidden_dim, aggregators=aggregators, scalers=scalers, deg=deg,
                          towers=5, pre_layers=1, post_layers=1, divide_input=False)
                self.convs.append(conv)
                self.grus.append(GRUCell(hidden_dim, hidden_dim))
                self.batch_norms.append(BatchNorm(hidden_dim))
                
        self.readout = PNAConv(in_channels=hidden_dim, out_channels=1, aggregators=aggregators, scalers=scalers, deg=deg,
                          towers=1, pre_layers=1, post_layers=1, divide_input=False)

        
    def forward(self, data):
        
        x, edge_index = data.x, data.edge_index
        for conv, gru, batch_norm in zip(self.convs, self.grus, self.batch_norms):
            y = conv(x, edge_index)
            x = gru(x, y)
            x = F.relu(batch_norm(x))
        x = self.readout(x, edge_index)

        return x

# Train function
def train(model, dataloader, optimizer, device):
    batch_loss = []
    model.train()
    
    for batch in dataloader:
        #node, edge, label = batch.x, batch.edge_index, batch.y
        #node = node.to(device)
        #edge = edge.to(device)
        #label = label.to(device)
        
        # Train the model on each batch
        label = torch.cat([data.y for data in batch]).to(device)
        pred = model(batch)
        loss = F.mse_loss(pred.squeeze(), label.squeeze())
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_loss.append(loss.item())
        
    return np.mean(np.array(batch_loss))

# Validation function
def validate(model, dataloader, device):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            #node, edge, label = batch.x, batch.edge_index, batch.y
            #node = node.to(device)
            #edge = edge.to(device)
            #label = label.to(device)
        
            # Validate the model on each batch
            label = torch.cat([data.y for data in batch]).to(device)
            pred = model(batch)
            loss = F.mse_loss(pred.squeeze(), label.squeeze())
            val_loss.append(loss.item())
    return np.mean(np.array(val_loss))

if __name__ == "__main__":
    # Read relevant data files
    f1 = open(args.data_path + "edge.txt", "r")
    f2 = open(args.data_path + "node_features.txt", "r")
    f3 = open(args.data_path + "node_labels.txt", "r")
    lines1 = f1.readlines()
    lines2 = f2.readlines()
    lines3 = f3.readlines()

    # Data preprocessing
    num_data = args.num_data
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
        if args.input_dim == 1:
          xs = [float(idx) for idx in lines2[i].split()[1:]]
          node_feature = [[xs[j]] for j in range(len(xs))]
        elif args.input_dim == 2:
          xs = [float(idx) for idx in lines2[2 * i].split()[1:]]
          ys = [float(idx) for idx in lines2[2 * i + 1].split()[1:]]
          node_feature = [[xs[j], ys[j]] for j in range(len(xs))]
        elif args.input_dim == 3:
          xs = [float(idx) for idx in lines2[3 * i].split()[1:]]
          ys = [float(idx) for idx in lines2[3 * i + 1].split()[1:]]
          zs = [float(idx) for idx in lines2[3 * i + 2].split()[1:]]
          node_feature = [[xs[j], ys[j], zs[j]] for j in range(len(xs))]  
        elif args.input_dim == 4:
          xs = [float(idx) for idx in lines2[4 * i].split()[1:]]
          ys = [float(idx) for idx in lines2[4 * i + 1].split()[1:]]
          zs = [float(idx) for idx in lines2[4 * i + 2].split()[1:]]
          ls = [float(idx) for idx in lines2[4 * i + 3].split()[1:]]
          node_feature = [[xs[j], ys[j], zs[j], ls[j]] for j in range(len(xs))]  
        else:
           raise Exception("Sorry, not available input dimension")  
      
        x = torch.tensor(node_feature, dtype=torch.float)
        node_label = [float(idx) * args.scale_factor for idx in lines3[i].split()[1:]]
        y = torch.tensor(node_label, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    mean_value = np.mean(np.array(ave))
    Train_data, Test_data = train_test_split(data_list, test_size = 0.2, random_state=42)
    Train_data, Val_data = train_test_split(Train_data, test_size = 0.125, random_state=42)

    batch_size = args.batch_size
    train_loader = DataListLoader(Train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataListLoader(Test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataListLoader(Val_data, batch_size=batch_size, shuffle=True)

    deg = torch.zeros(args.max_degree, dtype=torch.long)
    for data in Train_data:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    device = "cuda"
    torch.cuda.empty_cache()
    model = PNANet().to(device)
    model = DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20, min_lr=-1e-5, verbose=True)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Model architecture:")
    print(model)
    print("The number of trainable parameters is:{}".format(params))


    path = './ckpt/'
    # Training
    print("epoch", "train loss", "validation loss")

    val_loss_curve = []
    train_loss_curve = []

    for epoch in range(args.epoch):
    
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
