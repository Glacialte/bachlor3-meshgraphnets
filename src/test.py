import torch
import random
import numpy as np
import pandas as pd
import torch_scatter
import torch.nn as nn
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import DataLoader, Data
import time
import torch.optim as optim
from tqdm import trange
import copy
import matplotlib.pyplot as plt
import os
import torch
import h5py
import tensorflow.compat.v1 as tf
import functools
import json
import enum

datafile = '/root/datasets/test.h5'
data = h5py.File(datafile, 'r')
#Define the list that will return the data graphs
data_list = []

#define the time difference between the graphs
dt=0.01   #A constant: do not change!

#define the number of trajectories and time steps within each to process.
#note that here we only include 2 of each for a toy example.
number_trajectories = 100
number_ts = 10000

def triangles_to_edges(faces):
  """Computes mesh edges from triangles.
     Note that this triangles_to_edges method was provided as part of the
     code release for the MeshGraphNets paper by DeepMind, available here:
     https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
  """
  # collect edges from triangles
  edges = tf.concat([faces[:, 0:2],
                     faces[:, 1:3],
                     tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
  # those edges are sometimes duplicated (within the mesh) and sometimes
  # single (at the mesh boundary).
  # sort & pack edges as single tf.int64
  receivers = tf.reduce_min(edges, axis=1)
  senders = tf.reduce_max(edges, axis=1)
  packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
  # remove duplicates and unpack
  unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
  senders, receivers = tf.unstack(unique_edges, axis=1)
  # create two-way connectivity
  return (tf.concat([senders, receivers], axis=0),
          tf.concat([receivers, senders], axis=0))



class NodeType(enum.IntEnum):
    """
    Define the code for the one-hot vector representing the node types.
    Note that this is consistent with the codes provided in the original
    MeshGraphNets study: 
    https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9

with h5py.File(datafile, 'r') as data:
    for i,trajectory in enumerate(data.keys()):
        if(i==number_trajectories):
            break
        print("Trajectory: ",i)

        #We iterate over all the time steps to produce an example graph except
        #for the last one, which does not have a following time step to produce
        #node output values
        for ts in range(len(data[trajectory]['velocity'])-1):

            if(ts==number_ts):
                break

            #Get node features

            #Note that it's faster to convert to numpy then to torch than to
            #import to torch from h5 format directly
            momentum = torch.tensor(np.array(data[trajectory]['velocity'][ts]))
            #node_type = torch.tensor(np.array(data[trajectory]['node_type'][ts]))
            node_type = torch.tensor(np.array(tf.one_hot(tf.convert_to_tensor(data[trajectory]['node_type'][0]), NodeType.SIZE))).squeeze(1)
            x = torch.cat((momentum,node_type),dim=-1).type(torch.float)

            #Get edge indices in COO format
            edges = triangles_to_edges(tf.convert_to_tensor(np.array(data[trajectory]['cells'][ts])))

            edge_index = torch.cat( (torch.tensor(edges[0].numpy()).unsqueeze(0) ,
                        torch.tensor(edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

            #Get edge features
            u_i=torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))[edge_index[0]]
            u_j=torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))[edge_index[1]]
            u_ij=u_i-u_j
            u_ij_norm = torch.norm(u_ij,p=2,dim=1,keepdim=True)
            edge_attr = torch.cat((u_ij,u_ij_norm),dim=-1).type(torch.float)

            #Node outputs, for training (velocity)
            v_t=torch.tensor(np.array(data[trajectory]['velocity'][ts]))
            v_tp1=torch.tensor(np.array(data[trajectory]['velocity'][ts+1]))
            y=((v_tp1-v_t)/dt).type(torch.float)

            #Node outputs, for testing integrator (pressure)
            p=torch.tensor(np.array(data[trajectory]['pressure'][ts]))

            #Data needed for visualization code
            cells=torch.tensor(np.array(data[trajectory]['cells'][ts]))
            mesh_pos=torch.tensor(np.array(data[trajectory]['mesh_pos'][ts]))

            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,y=y,p=p,
                                cells=cells,mesh_pos=mesh_pos))

print("Done collecting data!")

print("Done saving data!")
print('dataset len: {}'.format(len(data_list)))

with h5py.File(datafile, 'r') as data:
    for i, trajectory in enumerate(data.keys()):
        print('i:{}, trajectory: {}'.format(i, trajectory))
