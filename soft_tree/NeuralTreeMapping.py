import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TreeMapping(nn.Module):

    def __init__(self,
                 depth: int):
        super(TreeMapping, self).__init__()
        self.depth = depth
        right_match = [0] * (2**(depth) - 1)
        left_match = [0] * (2**(depth) - 1)
        matches = []
        self.debug= False
        def recurse_(curr_depth, right_match_, left_match_, i):
            if curr_depth == depth:
                return [(right_match_, left_match_)]
            rm_copy_1 = right_match_.copy()
            lm_copy_1 = left_match_.copy()
            
            rm_copy_1[i] = 1
            lm_copy_1[i] = 0

            rm_copy_2 = right_match_.copy()
            lm_copy_2 = left_match_.copy()

            rm_copy_2[i] = 0
            lm_copy_2[i] = 1

            right_subtree_results = recurse_(curr_depth+1, rm_copy_1, lm_copy_1, 2*i+1)
            left_subtree_results = recurse_(curr_depth+1, rm_copy_2, lm_copy_2, 2*(i+1))
            return matches +  right_subtree_results + left_subtree_results

        matches = recurse_(0, right_match, left_match, 0)

        all_right_matches = np.array([matches[i][0] for i in range(len(matches))])
        all_left_matches = np.array([matches[i][1] for i in range(len(matches))])

        no_matches = np.ones(all_left_matches.shape)
        no_matches[all_left_matches == 1] = 0
        no_matches[all_right_matches == 1] = 0

        leaf_right_tens = torch.tensor(all_right_matches).float()
        leaf_left_tens = torch.tensor(all_left_matches).float()
        leaf_no_tens = torch.tensor(no_matches).float()


        left_matches = torch.zeros([2**(depth) - 1, 2**(depth) -1])
        right_matches = torch.zeros([2**(depth) - 1, 2**(depth) -1])
        no_matches = torch.ones([2**(depth) - 1, 2**(depth) -1])
        

        def node_recurse(idx, parent_idx, right_parent):
            if idx >= 2**(depth) - 1:
                return
            if parent_idx is not None:                
                # copy the parent's matches
                parent_right = right_matches[parent_idx].clone()
                parent_left = left_matches[parent_idx].clone()
                if right_parent:
                    parent_right[parent_idx] = 1
                    parent_left[parent_idx] = 0
                else:
                    parent_right[parent_idx] = 0
                    parent_left[parent_idx] = 1
                right_matches[idx] = parent_right
                left_matches[idx] = parent_left
            node_recurse(2*idx+1, idx, True)
            node_recurse(2*(idx+1), idx, False)


        node_recurse(0, None, False)

        no_matches[left_matches == 1] = 0
        no_matches[right_matches == 1] = 0

        decision_right_tens = right_matches.float()
        decision_left_tens = left_matches.float()
        decision_no_tens = no_matches.float()

        combined_right = torch.vstack([leaf_right_tens, decision_right_tens])
        combined_left = torch.vstack([leaf_left_tens, decision_left_tens])
        combined_no = torch.vstack([leaf_no_tens, decision_no_tens])
        self.left_bin_matches = nn.Parameter(combined_left, requires_grad=False)
        self.right_bin_matches = nn.Parameter(combined_right, requires_grad=False)
        self.no_bin_matches = nn.Parameter(combined_no, requires_grad=False)
        
        depth_mapping = []
        for i in range(depth):
            depth_mapping += [i] * (2**i)
        self.depth_mapping = nn.Parameter(torch.tensor(depth_mapping, dtype=torch.float32), requires_grad=False)
        self.max_leaves = 2**(depth)
        self.max_nodes = 2**(depth) - 1
        self.n_leaves = 2**(depth)
        self.n_nodes = 2**(depth) - 1
        self.pruned_leaves = []
        self.pruned_nodes = []

        self.path_matrix = nn.Parameter(torch.tensor(self.create_path_matrix()), requires_grad=False)

    def create_path_matrix(self):
        depth = self.depth
        # Total number of nodes in a perfect binary tree of given depth
        n_nodes = 2**depth - 1
        # Initialize an n_nodes x n_nodes matrix with zeros
        matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        
        # Populate the matrix according to the path relationship
        for j in range(1, n_nodes):
            # Track the path from the root to node j
            current = j
            while current > 0:
                parent = (current - 1) // 2  # Calculate the parent index
                matrix[parent, j] = 1  # Mark that parent is in the path to j
                current = parent
        
        return matrix

    def forward(self, x):
        right_probs = x
        left_probs = 1 - x

        right_matches = right_probs.unsqueeze(1) * self.right_bin_matches.unsqueeze(0)
        left_matches = left_probs.unsqueeze(1) * self.left_bin_matches.unsqueeze(0)
        

        comb = right_matches + left_matches + self.no_bin_matches

        path_prob = torch.prod(comb, dim=-1)
        # ^-- [batch_size, n_leaves + n_nodes]
        leaf_probs = path_prob[:, :self.n_leaves]
        node_probs = path_prob[:, self.n_leaves:]
        return leaf_probs, node_probs

