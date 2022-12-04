###############
### Part of this script is modified based on https://github.com/diozaka/diffseg ###
###############

import torch
import torch.nn as nn
import warnings
import math
import torch.nn.functional as F
from einops import rearrange
import sys
from torch_geometric import utils
# from torch_geometric.utils.unbatch import unbatch, unbatch_edge_index
from torch_geometric.nn import GraphConv, dense_mincut_pool
from einops import rearrange, repeat
# from libcpab.cpab import Cpab
from torch import Tensor

from torch_geometric.utils import degree
from typing import List
###############
### HELPERS ###
###############

def reset_parameters(m):
    if hasattr(m, "reset_parameters"):
        m.reset_parameters()

class Constant(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        self.output_sizes = output_sizes
        self.const = nn.Parameter(torch.Tensor(1, *output_sizes))

    # inp is an arbitrary tensor, whose values will be ignored;
    # output is self.const expanded over the first dimension of inp.
    # output.shape = (inp.shape[0], *output_sizes)
    def forward(self, inp):
        return self.const.expand(inp.shape[0], *((-1,)*len(self.output_sizes)))

    def reset_parameters(self):
        nn.init.uniform_(self.const, -1, 1) # U~[-1,1]

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
    def forward(self, inp):
        return inp.unsqueeze(self._dim)

class Square(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return inp*inp

class Abs(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.abs(inp)

class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.exp(inp)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inp):
        return torch.sin(inp)

#########################
### WARPING FUNCTIONS ###
#########################

class TSPStepWarp(nn.Module):
    def __init__(self, loc_net, width, power, min_step=0.0001, max_step=0.9999):
        # loc_net: nn.Module that takes shape (batch_size, seq_len, input_size)
        #          and produces shape (batch_size, n_seg-1) with logits that determine
        #          the modes of n_seg-1 TSP distributions.
        #          More precisely, modes for the TSP distributions are computed from
        #          the cumulative sum of the softmax over the logits. Logits thus
        #          encode the relative distances between consecutive modes.
        #          For numerical reasons, the modes are clamped to the
        #          interval [min_step, max_step].
        # width: width of the TSP distributions, from (0,1]
        # power: power of the TSP distributions, must be >=1 for unimodality

        super().__init__()
        if not isinstance(loc_net, nn.Module):
            raise ValueError("loc_net must be an instance of torch.nn.Module")

        self.loc_net = loc_net
        self.width = width
        self.power = power
        self.min_step = min_step
        self.max_step = max_step

    def _tsp_params(self, mode):
        # mode.shape = (n_modes,)
        # output.shape = ((n_modes, 1), (n_modes, 1), (n_modes, 1), double)
        a = torch.clamp(mode-self.width/2., 0., 1.-self.width).unsqueeze(1).to(mode.device) # max(0., min(1.-width, mode-width/2))
        b = torch.clamp(a+self.width, self.width, 1.).to(mode.device) # min(1., a+width)
        m = mode.unsqueeze(1) # [B*(n_seg-1)]
        n = self.power
        return a, b, m, n

    def _tsp_cdf(self, x, mode):
        # x.shape = (n_modes, seq_len)
        # mode.shape = (n_modes,)
        a, b, m, n = self._tsp_params(mode) # m: (n_modes,1)
        cdf = ((x <= m)*((m-a)/(b-a)*torch.pow(torch.clamp((x-a)/(m-a), 0., 1.).to(mode.device), n))
              +(m <  x)*(1.-(b-m)/(b-a)*torch.pow(torch.clamp((b-x)/(b-m), 0., 1.).to(mode.device), n)))
        return cdf # [B*(n_seg-1), seq_len]

    def forward(self, input_seq,mask):
        # input_seq.shape = (batch_size, seq_len, input_size)
        # output shape = (batch_size, seq_len)
        batch_size, seq_len, input_size = input_seq.shape

        # compute modes for all triangular mixture distributions from loc_net
        #modes = nn.functional.softmax(self.loc_net(input_seq), dim=1).cumsum(dim=1)[:,:-1] # last is always 1
        mu = self.loc_net(input_seq,mask)
        modes = nn.functional.softmax(torch.cat([torch.zeros((batch_size,1)).to(input_seq.device), # fix first logit to 0
                                                 mu], dim=1),
                                      dim=1).cumsum(dim=1)[:,:-1] # last boundary is always 1
        modes = torch.clamp(modes, self.min_step, self.max_step).to(input_seq.device) # [B,n_seg-1]
        _, n_steps = modes.shape # == n_seg-1

        xrange = torch.linspace(0, 1, seq_len).unsqueeze(0).expand(n_steps*batch_size,-1).to(modes.device) # [B*(n_seg-1), seq_len]
        cdf = self._tsp_cdf(xrange, modes.flatten())

        # compute mixture cdf
        gamma = cdf.reshape(-1,n_steps,seq_len).sum(dim=1)/n_steps #[b,seq_len]
        return gamma, n_steps+1, modes, mu


# backend can be any nn.Module that takes shape (batch_size, seq_len, input_size)
# and produces shape (batch_size, seq_len); the output of the backend is normalized
# and integrated.
class VanillaWarp(nn.Module):
    def __init__(self, backend, nonneg_trans='abs'):
        super().__init__()
        if not isinstance(backend, nn.Module):
            raise ValueError("backend must be an instance of torch.nn.Module")
        self.backend = backend
        self.normintegral = NormalizedIntegral(nonneg_trans)

    # input_seq.shape = (batch_size, seq_len, input_size)
    # output shape = (batch_size, seq_len)
    def forward(self, input_seq):
        gamma = self.normintegral(self.backend(input_seq)) 
        return gamma

class NormalizedIntegral(nn.Module):
    # {abs, square, relu}      -> warping variance more robust to input variance
    # {exp, softplus, sigmoid} -> warping variance increases with input variance, strongest for exp
    def __init__(self, nonneg):
        super().__init__()
        # higher warping variance
        if nonneg == 'square':
            self.nonnegativity = Square()
        elif nonneg == 'relu':
            warnings.warn('ReLU non-negativity does not necessarily result in a strictly monotonic warping function gamma! In the worst case, gamma == 0 everywhere.', RuntimeWarning)
            self.nonnegativity = nn.ReLU()
        elif nonneg == 'exp':
            self.nonnegativity = Exp()
        # lower warping variance
        elif nonneg == 'abs':
            self.nonnegativity = Abs()
        elif nonneg == 'sigmoid':
            self.nonnegativity = nn.Sigmoid()
        elif nonneg == 'softplus':
            self.nonnegativity = nn.Softplus()
        else:
            raise ValueError("unknown non-negativity transformation, try: abs, square, exp, relu, softplus, sigmoid")

    # input_seq.shape = (batch_size, seq_len)
    # output shape    = (batch_size, seq_len)
    def forward(self, input_seq):
        # transform sequences to alignment functions between 0 and 1
        dgamma = torch.cat([torch.zeros((1,1)), # fix entry to 0
                            self.nonnegativity(input_seq)], dim=1)
        gamma = torch.cumsum(dgamma, dim=1)
        gamma /= torch.max(gamma, dim=1)[0].unsqueeze(1)
        return gamma

##########################
### SEGMENTATION LAYER ###
##########################

class Cal_M(nn.Module):
    def __init__(self, dim, output_len):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Linear(dim, dim*2),
            nn.Tanh(),
            nn.Linear(dim*2,output_len, bias=False))

        self.reduce_d = nn.Linear(dim, 1)
    
    def forward(self, h, mask=None, mask_value=-1e30):
        # input: (batch_size, seq_len, input_size)
        # output: (batch_size, n_seg-1)

        attn = self.attn(h) # [B,L,L']
        if mask is not None:
            mask = rearrange(mask, 'b l -> b l 1')
            attn = mask * attn + (1-mask)*mask_value
        
        attn = F.softmax(attn, dim=-2) # [b,l,l']
        h = torch.matmul(h.transpose(-1, -2), attn) #.squeeze(-1) # [B,D L']
        h = self.reduce_d(rearrange(h, 'b d l -> b l d')).squeeze(-1) # [b l']
        # h = torch.tanh(h)
        return h
        

class TemSeg(nn.Module):
    def __init__(self, device, dim, K, width=0.125, power=16., min_step=0.0001, max_step=0.9999):
        super().__init__()
        
        m_enc = Cal_M(dim, K-1).to(device)
        self.warp = TSPStepWarp(m_enc, width=width, power=power, min_step=min_step, max_step=max_step).to(device)

    # gamma.shape    = (batch_size, warped_len), alignment functions mapping to [0,1]
    # original.shape = (batch_size, original_len, original_dim)
    # output.shape   = (batch_size, warped_len, original_dim)
    # almat.shape    = (batch_size, warped_len, original_len)
    def forward(self, input_seq, edge_weight, mask=None, kernel='linear'):
        # [B,L,D]
        gamma, original_len, modes, mu = self.warp(input_seq,mask)
        batch_size, warped_len = gamma.shape
        # _, original_len, original_dim = input_seq.shape
        gamma_scaled = gamma*(original_len-1)
        almat = torch.zeros(batch_size, warped_len, original_len)

        if kernel == 'integer':
            for k in range(original_len):
                responsibility = (torch.floor(gamma_scaled+0.5)-k == 0).float()
                almat[:,:,k] = responsibility
        elif kernel == 'linear':
            for k in range(original_len):
                responsibility = torch.threshold(1-torch.abs(gamma_scaled-k), 0., 0.)
                almat[:,:,k] = responsibility
        else:
            raise ValueError("unknown interpolation kernel, try 'integer' or 'linear'")
        
        input_seq, edge_weight = self.almat_aggregate(almat, input_seq, edge_weight)
        return almat, gamma_scaled, input_seq, edge_weight
    
    def almat_aggregate(self, almat, h0, edge_weight):
        # K: the number of clusters
        # event_time: [B,L]
        # h0: [B,K,L,D]
        # non_pad_mask: [B,K,L]

        b, t, dim = h0.shape
        _, _, new_t= almat.shape
        new_h0 = torch.zeros((b, new_t, dim)).to(h0.device)
        new_edge_weight = torch.zeros((b, new_t, edge_weight.shape[-1])).to(h0.device)

        # apply different segmentation for different modalities
        # h = rearrange(h0, 'b k l d -> (b k) l d')
        # len_mask = rearrange(non_pad_mask, 'b k l -> (b k) l')

        almat = almat.to(h0.device)
        
        # apply different segmentation for different modalities
        # almat = rearrange(almat, 'b s l -> b l s')
        
        # directly multiply h0 and alignment matrix
        # new_h0 = torch.matmul(rearrange(h0, 'b k t d -> b k d t'), rearrange(almat, 'b t l -> b 1 t l')) 
        # new_h0 = rearrange(new_h0, 'b k d l -> b k l d')
        
        # pooling each slots
        for i in range(new_t):
            # apply different segmentation for different modalities
            idx = almat[:,:,i]

            # weight
            tmp_h0 = h0 * idx.unsqueeze(-1) # [B,L,D]
            tmp_edge_weight = edge_weight * idx.unsqueeze(-1)
            # sum
            # new_h0[:,i,:] = torch.sum(tmp_h0, dim=1)
            
            
            # pooling
            tmp_h0 = rearrange(tmp_h0, 'b l d -> b d l')
            tmp_h0 = F.max_pool1d(tmp_h0, tmp_h0.size(-1)).squeeze() # [B d]
            new_h0[:,i,:] = tmp_h0
            
            tmp_edge_weight = rearrange(tmp_edge_weight, 'b l d -> b d l')
            tmp_edge_weight = F.max_pool1d(tmp_edge_weight, tmp_edge_weight.size(-1)).squeeze() # [B d]
            new_edge_weight[:,i,:] = tmp_edge_weight
            

        return new_h0, new_edge_weight

    
class GraphSeg(torch.nn.Module):
    def __init__(self, 
                 dim, 
                 n_clusters, 
                 mlp_units=[],
                 mlp_act="Identity"):
        super().__init__()

        mlp_act = getattr(torch.nn, mlp_act)(inplace=True)
        
        # MLP layers
        # self.mlp = torch.nn.Sequential()
        # for units in mlp_units:
        #     self.mlp.append(nn.Linear(dim, units))
        #     dim = units
        #     self.mlp.append(mlp_act)
        # self.mlp.append(nn.Linear(dim, n_clusters))
        self.mlp = nn.Linear(dim, n_clusters)
        

    def forward(self, x, edge_index, edge_weight):
        # Cluster assignments (logits)
    
        s = self.mlp(x) 
        b, v, l = s.shape
        
        edge_index_list = []
        edge_attr_list = []
        x_list = []
        total_mc_loss, total_o_loss = 0, 0
        
        # Obtain MinCutPool losses
        for i in range(b):
            if isinstance(edge_index, list):
                adj = utils.to_dense_adj(edge_index[i], edge_attr=edge_weight[i], max_num_nodes=v)
            else:
                adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=v)
            # batch = torch.tensor([[int(i)]*l for i in range(b)]).reshape(-1)
            
            out, out_adj, mc_loss, o_loss = dense_mincut_pool(x[i], adj[0], s[i])
            tmp_edge_index, tmp_edge_weight = utils.dense_to_sparse(out_adj)
            if len(tmp_edge_weight) > 0:
                edge_index_list.append(tmp_edge_index.unsqueeze(0))
                edge_attr_list.append(tmp_edge_weight.unsqueeze(0))
                x_list.append(out)
                total_mc_loss = total_mc_loss + mc_loss
                total_o_loss = total_o_loss + o_loss
            
        x_list = torch.cat(x_list, dim=0)
        try:
            edge_attr_list = torch.cat(edge_attr_list, dim=0)
        except:
            print("error occured")
            for attr in edge_attr_list:
                print(attr.shape)
            sys.exit(0)
        edge_index_list = torch.cat(edge_index_list, dim=0)
        return torch.softmax(s, dim=-1), x_list, edge_index_list, edge_attr_list, total_mc_loss, total_o_loss


    
class TemPool(nn.Module):
    def __init__(self, device, dim, v_num_list, t_num_list):
        super().__init__()
        
        self.graph_seg_stack = [
            GraphSeg(dim, new_l).to(device)
            for new_l in v_num_list]
       
        self.tem_seg_stack = [
            TemSeg(device, dim*v_num_list[i], t_num_list[i], width=(1/t_num_list[i])*2, power=16, min_step=0.0001, max_step=0.9999) #.to(device)
            for i in range(len(t_num_list))]
        


    def forward(self, idx, X, edge_index, edge_weight, viz=False):
        # (B, N, D, T)
        b, n, d ,t = X.shape
        X = rearrange(X, 'b n d t -> (b t) n d')
        
        # X:(B*T,N,D) -> (B*T,N_new, D)
        s_graph, X, edge_index, edge_weight, mc_loss, o_loss = self.graph_seg_stack[idx](X, edge_index, edge_weight) 
        
        X = rearrange(X, '(b t) n d -> b t (n d)', b=b)

        if not viz:
            s_time, _, X, edge_weight = self.tem_seg_stack[idx](X, edge_weight) # (B,T,N*D) => (B,T_new, N*D)
        else:
            s_time, _, X, edge_weight = self.tem_seg_stack[idx](X, edge_weight, kernel='integer')
            
        X = rearrange(X, 'b t (n d) -> b n d t', d=d)

        edge_index = rearrange(edge_index, '(b t) i m ->b i m t', b=b)
        edge_weight = rearrange(edge_weight, 'b t m -> b m t') 
        
        
        if not viz:
            return X, edge_index, edge_weight, mc_loss, o_loss
        else:
            return X, edge_index, edge_weight, mc_loss, o_loss, s_graph, s_time


