import pickle as pkl
import numpy as np
import scipy.sparse as sp
import sys
import networkx as nx


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



dataset_str = "cora"

"""Load data."""
names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
objects = []
for i in range(len(names)):
    with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
        if sys.version_info > (3, 0):
            objects.append(pkl.load(f, encoding='latin1'))
        else:
            objects.append(pkl.load(f))

x, y, tx, ty, allx, ally, graph = tuple(objects)

# print()
# print(graph)


test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
test_idx_range = np.sort(test_idx_reorder)

features = sp.vstack((allx, tx)).tolil()

features[test_idx_reorder, :] = features[test_idx_range, :]
adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

# print(adj)

labels = np.vstack((ally, ty))
labels[test_idx_reorder, :] = labels[test_idx_range, :]

print(features.shape)
print(labels.shape)
# print(labels)
# print(graph[0])
print(adj.shape)
print(type(adj))

idx_test = test_idx_range.tolist()
idx_train = range(len(y))
idx_val = range(len(y), len(y)+500)

train_mask = sample_mask(idx_train, labels.shape[0])
val_mask = sample_mask(idx_val, labels.shape[0])
test_mask = sample_mask(idx_test, labels.shape[0])

# print(train_mask)

y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)
y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]






#########

import json


# infile = open("ppi/ppi-class_map.json")
# tmp = json.load(infile)
# print(len(tmp['0']))
# infile.close()

# infile = open("ppi/ppi-G.json")
# tmp = json.load(infile)
# print(tmp.keys())

# test_cnt = 0
# val_cnt = 0

# for xxx in tmp['nodes']:
#     if xxx['test'] == True:
#         test_cnt += 1
#     if xxx['val'] == True:
#         val_cnt += 1

# print(test_cnt, val_cnt)

# print(tmp['nodes'][1])
# # print(len(tmp['links']))
# infile.close()



# with open("ppi/ppi-feats.npy", 'rb') as f:
#     x_in = np.load(f)
#     print(x_in.shape)


#########################

with open("ppi/ppi-feats.npy", 'rb') as fin:
    features = np.load(fin)
    print(features.shape)

n_node = features.shape[0]  # 56944
n_feature = features.shape[1]  # 50
n_class = 121  # 121

idx_test = []
idx_train = []
idx_val = []

infile = open("ppi/ppi-G.json")

for index, one in enumerate(json.load(infile)['nodes']):
    if one['test'] == True:
        idx_test.append(one['id'])
    if one['val'] == True:
        idx_val.append(one['id'])
    if one['test'] == False and one['val'] == False:
        idx_train.append(one['id'])

print(len(idx_test))
print(len(idx_val))
print(len(idx_train))

train_mask = sample_mask(idx_train, n_node)
val_mask = sample_mask(idx_val, n_node)
test_mask = sample_mask(idx_test, n_node)

#------------------------------------------

labels = np.zeros((n_node, n_class))

with open("ppi/ppi-class_map.json", 'r') as fin:
    label_json = json.load(fin)
    for idx in range(n_node):
        labels[idx] = label_json[str(idx)]

y_train = np.zeros(labels.shape)
y_val = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)

y_train[train_mask, :] = labels[train_mask, :]
y_val[val_mask, :] = labels[val_mask, :]
y_test[test_mask, :] = labels[test_mask, :]

#------------------------------------------

graph_dict = {}
with open("ppi/ppi-walks.txt", 'r') as fin:
    for oneline in fin.readlines():
        one_list = oneline.rstrip('\n').split('\t')
        left = int(one_list[0])
        right = int(one_list[1])

        if left not in graph_dict:
            graph_dict[left] = [right]
        else:
            graph_dict[left].append(right)

adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))

print(adj.shape)

