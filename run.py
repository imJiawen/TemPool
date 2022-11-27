try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable):
        return iterable
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from util import util
from einops import rearrange, repeat
import os

from torch_geometric_temporal.nn.recurrent import TGCN, A3TGCN, AGCRN, DCRNN, EvolveGCNH, LRGCN, MPNNLSTM
from torch_geometric_temporal.nn.attention import ASTGCN, STConv, MSTGCN
from model import PoolASTGCN

from torch_geometric_temporal.dataset import WindmillOutputLargeDatasetLoader, WindmillOutputMediumDatasetLoader, WindmillOutputSmallDatasetLoader
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, TwitterTennisDatasetLoader, EnglandCovidDatasetLoader, MontevideoBusDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import sys
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", type=str, default="/home/covpreduser/Blob/")
parser.add_argument('--model_name', type=str, default='poolastgcn')
parser.add_argument('--dataset', type=str, default='twi')
parser.add_argument('--epoch_num', type=int, default=1)
parser.add_argument('--use_wandb', type=str, default=None)
opt = parser.parse_args()

seed = 1
es_num = 5
# GPU support
DEVICE = torch.device('cuda') # cuda
util.setup_seed(seed)
early_stop = util.Early_Stop(es_num)

model_name = opt.model_name # ['astgcn','a3tgcn2', 'agcrn','dcrnn', 'stconv', 'mstgcn', 'tgcn', 'evolgcn]
dataset = opt.dataset # ['mtr','encov', 'twi','mtm']
epoch_num = opt.epoch_num

tempool = [[0.5,0.25], [0.5,0.25]]

if model_name in ['mstgcn', 'lrgcn','evolgcn','stconv','agcrn']:
    lr = 1e-3
elif model_name in ['a3tgcn2','tgcn','astgcn', 'poolastgcn']:
    lr = 1e-4
elif model_name in ['dcrnn','mpnnlstm']:
    lr = 1e-2

if dataset == 'ckp':
    loader = ChickenpoxDatasetLoader()
    in_step = 8
    out_step = 1
    num_of_vertices = 20
    node_features = 1
elif dataset == 'wind_m':
    loader = WindmillOutputMediumDatasetLoader()
    in_step = 8
    out_step = 1
    num_of_vertices = 26
    node_features = 1
elif dataset == 'wind_s':
    loader = WindmillOutputSmallDatasetLoader()
    in_step = 8
    out_step = 1
    num_of_vertices = 11
    node_features = 1
elif dataset == 'wind_l':
    loader = WindmillOutputLargeDatasetLoader()
    in_step = 8
    out_step = 1
    num_of_vertices = 319
    node_features = 1
elif dataset == 'eng':
    loader = EnglandCovidDatasetLoader()
    in_step = 8
    out_step = 1
    num_of_vertices = 129
    node_features = 1
elif dataset == 'twi_rg17':
    num_of_vertices = 1000
    loader = TwitterTennisDatasetLoader(event_id='rg17',N=num_of_vertices,target_offset=1)
    in_step = 16
    out_step = 1
    node_features = 1
    tempool = [[0.2,0.1], [0.5, 0.25]]
elif dataset == 'twi_uo17':
    num_of_vertices = 1000
    loader = TwitterTennisDatasetLoader(event_id='uo17',N=num_of_vertices,target_offset=1)
    in_step = 16
    out_step = 1
    node_features = 1
    tempool = [[0.2,0.1], [0.5, 0.25]]
elif dataset == 'bus':
    loader = MontevideoBusDatasetLoader()
    in_step = 8
    out_step = 1
    num_of_vertices = 675
    node_features = 1
    tempool = [[0.2,0.05], [0.5, 0.25]]

if model_name in ['stconv', 'agcrn']:
    hid_dim = 4
    if model_name == 'agcrn':
        e = torch.empty(num_of_vertices, hid_dim).to(DEVICE)
        torch.nn.init.xavier_uniform_(e)

batch_size = 1

"""
## Creating data loader
"""

if dataset in ['ckp', 'bus']:
    dataset = loader.get_dataset(in_step)
else:
    dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)

b_train = len(train_dataset.features)
b_test = len(test_dataset.features)

if opt.use_wandb is not None:
    os.environ['WANDB_SILENT']="true"
    wandb.login(key=str('ba7fa76ce534149710bbbc32b6fb8363c4f13044'))
    wandb.init(
        name= model_name+'_'+ opt.dataset,
        project=opt.use_wandb, 
        entity="imjiawen",
        tags=[opt.dataset, model_name, 'epoch'+str(opt.epoch_num)],
        dir=opt.root_dir + "v-jiawezhang/wandb/",
        config = opt)

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods, batch_size, out_dim=32, num_of_vertices=None):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        if model_name == 'a3tgcn2':
            self.tgnn = A3TGCN(in_channels=node_features, out_channels=out_dim, periods=periods)#,batch_size=batch_size) # node_features=2, periods=12
        elif model_name == 'dcrnn':
            self.tgnn = DCRNN(node_features*in_step, out_dim, 1)
        elif model_name == 'stconv':
            self.tgnn = STConv(num_nodes=num_of_vertices, in_channels=node_features, hidden_channels=hid_dim, out_channels=16, kernel_size=2, K=3)
            out_dim = 16*(in_step-2)
        if model_name == 'tgcn':
            self.tgnn = TGCN(in_step*node_features, out_dim) #,batch_size=batch_size, add_self_loops=False
            periods = 1
        if model_name == 'evolgcn':
            self.tgnn = EvolveGCNH(num_of_nodes=num_of_vertices, in_channels=node_features*in_step)
            out_dim = node_features*in_step
        if model_name == 'mpnnlstm':
            self.tgnn = MPNNLSTM(in_channels=node_features*in_step, hidden_size=out_dim, num_nodes=num_of_vertices, window=out_step, dropout=0.5)
            out_dim = 2*out_dim + in_step
            periods = out_step
            
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(out_dim, periods)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        if edge_weight is not None:
            h = self.tgnn(x, edge_index, edge_weight) 
        else:
            h = self.tgnn(x, edge_index)

        if model_name == 'stconv':
            h = rearrange(h, 'b t n f -> b n (t f)')

        h = F.relu(h) 
        h = self.linear(h)
        
        return h
    
    
class RecurrentGCN(torch.nn.Module):
    def __init__(self, node_features, periods, out_dim=4, num_of_vertices=None):
        super(RecurrentGCN, self).__init__()
        self.tgnn = AGCRN(number_of_nodes = num_of_vertices,
                                in_channels = node_features*in_step,
                                out_channels =out_dim,
                                K = 2,
                                embedding_dimensions=hid_dim)
        self.linear = torch.nn.Linear(out_dim, periods)

    def forward(self, x, e, h):
        h_0 = self.tgnn(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0
    
class RunLRGCN(torch.nn.Module):
    def __init__(self, node_features, periods, out_dim=32):
        super(RunLRGCN, self).__init__()
        self.tgnn = LRGCN(in_channels=node_features*in_step, out_channels=out_dim, num_relations=1, num_bases=1)
        self.linear = torch.nn.Linear(out_dim, periods)

    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h_0, c_0 = self.tgnn(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h_0)
        h = self.linear(h)
        return h, h_0, c_0
    
cml_loss = True
if model_name == 'poolastgcn':
    cml_loss = False
    print("tempool: ", tempool)
    model = PoolASTGCN(in_channels=node_features, len_input=in_step, nb_block=3, K=3,time_strides=1,\
                nb_chev_filter=32, nb_time_filter=32, num_for_predict=out_step, num_of_vertices=num_of_vertices, out_dim=1, tempool=tempool, device=DEVICE).to(DEVICE)
elif model_name == 'astgcn':
    cml_loss = False
    model = ASTGCN(in_channels=node_features, len_input=in_step, nb_block=3, K=3,time_strides=1,\
                nb_chev_filter=32, nb_time_filter=32, num_for_predict=out_step, num_of_vertices=num_of_vertices, normalization="sym").to(DEVICE) #
elif model_name == 'mstgcn':
    cml_loss = False
    model = MSTGCN(in_channels=node_features, len_input=in_step, nb_block=3, K=3,time_strides=1,\
                nb_chev_filter=32, nb_time_filter=32, num_for_predict=out_step).to(DEVICE)
elif model_name in ['agcrn']:
    model = RecurrentGCN(node_features=node_features, periods=out_step, num_of_vertices=num_of_vertices).to(DEVICE)
elif model_name in ['lrgcn']:
    model = RunLRGCN(node_features=node_features, periods=out_step).to(DEVICE)
else:
    model = TemporalGNN(node_features=node_features, periods=out_step, batch_size=batch_size, num_of_vertices=num_of_vertices).to(DEVICE)
    
print(model)
print(opt)
print("lr: ", str(lr))

model = model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if opt.use_wandb is not None:
    wandb.watch(model)
    
model.train()

for epoch in tqdm(range(epoch_num)):
    total_loss = []
    cost = 0
    h1 = None
    time = 0
    h, c = None, None
    # for encoder_inputs, labels in tqdm(train_loader, mininterval=2, desc='  - (Training)   ', leave=False):
    for snapshot in tqdm(train_dataset, total=b_train, mininterval=2, desc='  - (Training)   ', leave=False):
        x = snapshot.x.to(DEVICE)
        if model_name in ['poolastgcn']:
            x = rearrange(x, 'n t -> 1 n 1 t').to(DEVICE)
            y_hat, total_mc_loss, total_o_loss = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE))
            cost = cost + total_mc_loss + total_o_loss
        elif model_name in ['astgcn', 'mstgcn']:
            x = rearrange(x, 'n t -> 1 n 1 t').to(DEVICE)
            y_hat = model(x, snapshot.edge_index.to(DEVICE))
        elif model_name in ['a3tgcn2']:
            x = rearrange(x, 'n t -> n 1 t').to(DEVICE)
            y_hat = model(x, snapshot.edge_index.to(DEVICE))
        elif model_name in ['tgcn','dcrnn', 'evolgcn', 'mpnnlstm']:
            y_hat = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE))
        elif model_name in ['agcrn']:
            x = rearrange(x, 'n t -> 1 n t').to(DEVICE)
            y_hat, h1 = model(x, e, h1)
        elif model_name == 'stconv':
            x = rearrange(x, 'n t -> 1 t n 1').to(DEVICE)
            y_hat = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE))
        elif model_name == 'lrgcn':
            y_hat, h, c = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE), h, c)
        
        cost = cost + torch.mean((y_hat-snapshot.y.to(DEVICE))**2)
        if cml_loss:
            time += 1
        else:
            cost.backward()
            total_loss.append(cost.item())
            cost = 0

    if cml_loss:
        cost = cost / (time+1)
        cost.backward()
        total_loss = cost.item()
    else:
        total_loss = np.average(total_loss)
        
    if opt.use_wandb is not None:
        wandb.log({"train_loss": total_loss})
        
    optimizer.step()
    optimizer.zero_grad()
    
    
    # early stop
    end_flag = early_stop(total_loss,model)
    if end_flag:
        break
    
    print("\nLOSS: {:.4f}".format(total_loss))
    
model.eval()
mse_list = []
rmse_list = []
print("loading model...")
model.load_state_dict(early_stop.model_state_dict)
h, c = None, None
for snapshot in tqdm(test_dataset, total=b_test, mininterval=2, desc='  - (Testing)   ', leave=False):
    x = snapshot.x.to(DEVICE)
    if model_name in ['poolastgcn']:
        x = rearrange(x, 'n t -> 1 n 1 t').to(DEVICE)
        y_hat, _, _ = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE))
    elif model_name in ['astgcn', 'mstgcn']:
        x = rearrange(x, 'n t -> 1 n 1 t').to(DEVICE)
        y_hat = model(x, snapshot.edge_index.to(DEVICE))
    elif model_name in ['a3tgcn2']:
        x = rearrange(x, 'n t -> n 1 t').to(DEVICE)
        y_hat = model(x, snapshot.edge_index.to(DEVICE))
    elif model_name in ['tgcn','dcrnn', 'evolgcn', 'mpnnlstm']:
        y_hat = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE))
    elif model_name in ['agcrn']:
        x = rearrange(x, 'n t -> 1 n t').to(DEVICE)
        y_hat, h1 = model(x, e, h1)
    elif model_name == 'stconv':
        x = rearrange(x, 'n t -> 1 t n 1').to(DEVICE)
        y_hat = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE))
    elif model_name == 'lrgcn':
        y_hat, h, c = model(x, snapshot.edge_index.to(DEVICE), snapshot.edge_attr.to(DEVICE), h, c)
        
    mse_list.append(torch.mean((y_hat-snapshot.y.to(DEVICE))**2).item())
    rmse_list.append(np.sqrt(torch.mean((y_hat-snapshot.y.to(DEVICE))**2).item()))


mse = np.average(mse_list)
rmse = np.average(rmse_list)
print("MSE: {:.4f}".format(mse))
print("RMSE: {:.4f}".format(rmse))

if opt.use_wandb is not None:
    results = [[opt.model_name, opt.dataset, mse, rmse]]
    wandb.log({"Test Results": wandb.Table(data=results,
                                columns = ["model", "dataset", "mse", "rmse"])})
    wandb.finish()