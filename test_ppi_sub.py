import pickle
import numpy as np
import scipy.sparse as sp
import sys
import networkx as nx
import json

#------------------------------------------
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

#------------------------------------------
with open("new_ppi/feature_sub.npy", 'rb') as fin:
    features = np.load(fin)
# infile = open("new_ppi/feature_sub.pkl", 'rb')
# features = pickle.load(infile)
# infile.close()

print(features.shape)

n_node = features.shape[0]  # 5186
n_feature = features.shape[1]  # 50
n_class = 121  # 121

#------------------------------------------
idx_test = []
idx_train = []
idx_val = []

with open("new_ppi/train_ids.txt", 'r') as infile:
    for oneline in infile.readlines():
        idx = int(oneline.rstrip('\n'))
        idx_train.append(idx)

with open("new_ppi/valid_ids.txt", 'r') as infile:
    for oneline in infile.readlines():
        idx = int(oneline.rstrip('\n'))
        idx_val.append(idx)

with open("new_ppi/test_ids.txt", 'r') as infile:
    for oneline in infile.readlines():
        idx = int(oneline.rstrip('\n'))
        idx_test.append(idx)

print(len(idx_train))
print(len(idx_val))
print(len(idx_test))

train_mask = sample_mask(idx_train, n_node)
val_mask = sample_mask(idx_val, n_node)
test_mask = sample_mask(idx_test, n_node)

#------------------------------------------
# infile = open("new_ppi/label_sub.pkl", 'rb')
# labels = pickle.load(infile)
# infile.close()

with open("new_ppi/label_sub.npy", 'rb') as fin:
    labels = np.load(fin)

y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)

y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]

#------------------------------------------
graph_dict = {}
for idx in range(len(idx_train) + len(idx_val) + len(idx_test)):
    graph_dict[idx] = []

with open("new_ppi/adj_sub.txt", 'r') as fin:
    for oneline in fin.readlines():
        one_list = oneline.rstrip('\n').split('\t')
        left = int(one_list[0])
        right = int(one_list[1])

        graph_dict[left].append(right)
        graph_dict[right].append(left)

adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
print(adj.shape)
print(type(adj))