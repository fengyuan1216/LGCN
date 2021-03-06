import os
import sys
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from networkx.readwrite import json_graph


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


def load_data(dataset_str):
    return load_small_data(dataset_str)


def load_cora_data(dataset_str):
    """Load cora data."""
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_ppi_sub_data():
    with open("new_ppi/feature_sub.npy", 'rb') as fin:
        features = np.load(fin)

    print(features.shape)

    n_node = features.shape[0]  # 5186
    n_feature = features.shape[1]  # 50
    n_class = 121  # 121

    #------------------------------------------
    idx_train = []
    idx_val = []
    idx_test = []

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

    idx_train = [i for i in range(0, 1000, 1)]
    idx_val = [i for i in range(1000, 1300, 1)]
    idx_test = [i for i in range(1300, 1500, 1)]

    print(len(idx_train))
    print(len(idx_val))
    print(len(idx_test))

    train_mask = sample_mask(idx_train, n_node)
    val_mask = sample_mask(idx_val, n_node)
    test_mask = sample_mask(idx_test, n_node)

    #------------------------------------------
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
    for idx in range(n_node):
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
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_ppi_data():
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

    try:
        fin = open("ppi/ppi-walks.pkl", 'rb')
        graph_dict = pkl.load(fin)
        print("load from pkl file")
    except:
        graph_dict = {}
        for idx in range(len(idx_train)):
            graph_dict[idx] = []

        with open("ppi/ppi-walks.txt", 'r') as fin:
            for oneline in fin.readlines():
                one_list = oneline.rstrip('\n').split('\t')
                left = int(one_list[0])
                right = int(one_list[1])

                graph_dict[left].append(right)
                graph_dict[right].append(left)

        fout = open("ppi/ppi-walks.pkl", 'wb')
        pkl.dump(graph_dict, fout)
        fout.close()

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph_dict))
    print(adj.shape)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_small_data(dataset_str):
    dataset_str = "ppi_sub"
    if dataset_str == "cora":
        return load_cora_data(dataset_str)
    elif dataset_str == "ppi":
        return load_ppi_data()
    else:
        return load_ppi_sub_data()


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    #import ipdb; ipdb.set_trace()
    try:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
    except Exception as e:
        print(e)
    
    if sparse:
        return sparse_to_tuple(features)
    # return features.todense()
    return features


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, norm=True, sparse=False):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj = adj + sp.eye(adj.shape[0])
    if norm:
        adj = normalize_adj(adj)
    if sparse:
        return sparse_to_tuple(adj)
    return adj.todense()


def construct_feed_dict(features, support, labels, labels_mask, is_training, placeholders):
    """Construct feed dictionary."""
    feed_dict = {
        placeholders['labels']: labels,
        placeholders['labels_mask']: labels_mask,
        placeholders['features']: features,
        placeholders['support']: support,
        placeholders['num_features_nonzero']: features.shape,
        placeholders['is_training']: is_training}
    return feed_dict
