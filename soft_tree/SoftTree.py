import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from soft_tree.NeuralTreeMapping import TreeMapping
import pandas as pd
from entmax import Entmax15
import queue
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm, trange
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin



class NeuralTreeModule(nn.Module):
    def __init__(self, depth: int, input_dim: int, output_dim: int):
        super(NeuralTreeModule, self).__init__()

        # tree mapping matrix
        if depth < 1:
            raise ValueError("Depth must be at least 1.")
        self.tree = TreeMapping(depth) # only supports full binary trees
        self.n_nodes = 2**depth - 1
        self.n_leaves = 2**(depth)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        self.linear = nn.Linear(input_dim, self.n_nodes)
        self.leaf_weights = nn.Parameter(torch.randn(self.n_leaves, output_dim), requires_grad=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X, pred=False):
        # X: batch_size, input_dim

        branching_logits = self.linear(X) # batch_size, n_nodes
        soft_branching = self.sigmoid(branching_logits)
        leaf_probs, _ = self.tree(soft_branching)
        if pred:
            output = torch.matmul(leaf_probs, self.leaf_weights)
            return output, leaf_probs
        return leaf_probs
    def set_leaves(self, leaf_probs, y_train):
        # leaf_probs: n_samples, n_leaves
        # y_train: n_samples, output_dim
        # we want to set self.n_leaves to be the average of y_train for each leaf
        # we can do this by multiplying leaf_probs.T with y_train and dividing by the sum of leaf_probs.T
        with torch.no_grad():
            leaf_sums = torch.sum(leaf_probs, dim=0) # n_leaves
            leaf_sums[leaf_sums == 0] = 1 # to avoid division by zero
            new_leaves = torch.matmul(leaf_probs.T, y_train) / leaf_sums.unsqueeze(1) # n_leaves, output_dim
            self.leaf_weights.copy_(new_leaves)
