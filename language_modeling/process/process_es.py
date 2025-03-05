import torch
from itertools import chain
from model.layer.layers_v1 import split_tensor_along_last_dim


def process_X(X, ids, n_expert):
    idx_list = [torch.eq(ids, i).nonzero(as_tuple=True)[0] for i in range(n_expert)]
    # print(idx_list, "idx_list")
    # sys.exit(0)
    flattened_list = list(chain(*idx_list))

    X = torch.index_select(X, 0, torch.tensor(flattened_list).cuda())

    sub_lengths = [len(sublist) for sublist in idx_list]
    X = torch.split(X, sub_lengths, dim=0)
    X = [sub_tensor if sub_length != 0 else [] for sub_tensor, sub_length in zip(X, sub_lengths)]

    return X, idx_list

def preprocess(X, route_prob, k=2):

    batch_size, n_expert = route_prob.shape
    values, seq_ids = torch.topk(route_prob, k, dim=-1)
    X_, idx_list = process_X(X, seq_ids, n_expert)

    return X_, idx_list

def process_X2(X, ids, n_expert):
    idx_list = [torch.eq(ids, i).nonzero(as_tuple=True)[0] for i in range(n_expert)]
    X_ = []
    for i in range(n_expert):
        if (len(idx_list[i]) != 0):

            x = torch.index_select(X[i], 0, torch.tensor(idx_list[i]).cuda())
            X_.append(x)
        else:
            X_.append([])

    return X_, idx_list

def preprocess2(X, route_prob, k=2):
    batch_size, n_expert = route_prob.shape

    X = split_tensor_along_last_dim(X, n_expert)

    values, seq_ids = torch.topk(route_prob, k, dim=-1)
    X_, idx_list = process_X2(X, seq_ids, n_expert)

    return X_, idx_list


def process_X_MH(X, ids, n_expert):
    idx_list = [torch.eq(ids, i).nonzero(as_tuple=True)[0] for i in range(n_expert)]
    # print(idx_list, "idx_list")
    # sys.exit(0)
    flattened_list = list(chain(*idx_list))
    X = torch.index_select(X, 0, torch.tensor(flattened_list).cuda())

    sub_lengths = [len(sublist) for sublist in idx_list]
    X = torch.split(X, sub_lengths, dim=0)
    X = [sub_tensor if sub_length != 0 else [] for sub_tensor, sub_length in zip(X, sub_lengths)]
    # for i in X:
    #     print(i.shape, "sdksmieofj")
    # sys.exit(0)
    return X, idx_list

def preprocessMH(X, route_prob, k=2):

    batch_size, n_expert = route_prob.shape
    values, seq_ids = torch.topk(route_prob, k, dim=-1)
    X_, idx_list = process_X_MH(X, seq_ids, n_expert)

    return X_, idx_list
