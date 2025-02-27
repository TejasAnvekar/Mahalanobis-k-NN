import torch
from torch import nn
import random
import numpy as np
from data import load_data, load_ScanObjectNN
from model import DCP
from model_mah import DCP as MDCP


import argparse
import sklearn
from sklearn.svm import SVC
import glob
import torch.nn.functional as F
from tqdm import tqdm 

parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
# parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
#                     help='Dimension of embeddings')
parser.add_argument('--k', type=int, default=5, metavar='N',
                        help='Num of nearest neighbors to use')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'scanobjectnn'],
                        help='Dataset to evaluate')
parser.add_argument('--n_runs', type=int, default=10,
                        help='Num of few-shot runs')
parser.add_argument('--k_way', type=int, default=5,
                        help='Num of classes in few-shot')
parser.add_argument('--m_shot', type=int, default=10,
                        help='Num of samples in one class')
parser.add_argument('--n_query', type=int, default=20,
                        help='Num of query samples in one class')
parser.add_argument('--model_path', type=str, default='D:\\Codes\\dcp\\Checkpoints\\mah_dcp_v2\\models\\model.best.t7', metavar='N',
                        help='Pretrained model path')
parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Embedding nn to use, [pointnet, dgcnn]')
parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                    choices=['identity', 'transformer'],
                    help='Attention-based pointer generator to use, [identity, transformer]')
parser.add_argument('--head', type=str, default='svd', metavar='N',
                    choices=['mlp', 'svd', ],
                    help='Head to use, [mlp, svd]')
parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                    help='Dimension of embeddings')
parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                    help='Num of blocks of encoder&decoder')
parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                    help='Num of heads in multiheadedattention')
parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                    help='Num of dimensions of fc in transformer')
parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                    help='Dropout ratio in transformer')

parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp', 'mdcp'],
                        help='Dataset to model')
args = parser.parse_args()

device = torch.device("cuda")

#Try to load models

if args.model.lower() == "dcp":
    model = DCP(args).to(device)
elif args.model.lower() == "mdcp":
    model = MDCP(args).to(device)

    
state_dict = torch.load(args.model_path,map_location='cuda')

# model = nn.DataParallel(model).to(device)
new_state_dict = {}
for key in state_dict:
  new_key = key.replace('module.','')
  new_state_dict[new_key] = state_dict[key]

model.load_state_dict(new_state_dict)

print("Model Loaded !!")

if args.dataset == 'modelnet40':
    # ModelNet40 - Few Shot Learning

    data_train, label_train = load_data('train')
    data_test, label_test = load_data('test')
    n_cls = 40
    
elif args.dataset == 'scanobjectnn':
    # ScanObjectNN - Few Shot Learning

    data_train, label_train = load_ScanObjectNN('train')
    data_test, label_test = load_ScanObjectNN('test')
    n_cls = 15

label_idx = {}
for key in range(n_cls):
    label_idx[key] = []
    for i, label in enumerate(label_train):
        # if label[0] == key:
        if label == key:
            label_idx[key].append(i)

acc = []
print(f"Dataset: {args.dataset}, K_ways: {args.k_way}, M_shots: {args.m_shot}, N_Querys: {args.n_query}")
for run in tqdm(range(args.n_runs)):
    k = args.k_way ; m = args.m_shot ; n_q = args.n_query

    k_way = random.sample(range(n_cls), k)

    data_support = [] ; label_support = [] ; data_query = [] ; label_query = []
    for i, class_id in enumerate(k_way):
        support_id = random.sample(label_idx[class_id], m)
        query_id = random.sample(list(set(label_idx[class_id]) - set(support_id)), n_q)

        pc_support_id = data_train[support_id]
        pc_query_id = data_train[query_id]
        data_support.append(pc_support_id)
        label_support.append(i * np.ones(m))
        data_query.append(pc_query_id)
        label_query.append(i * np.ones(n_q))

    data_support = np.concatenate(data_support)
    label_support = np.concatenate(label_support)
    data_query = np.concatenate(data_query)
    label_query = np.concatenate(label_query)

    feats_train = []
    labels_train = []
    model = model.eval()

    for i in range(k * m):
        data = torch.from_numpy(np.expand_dims(data_support[i], axis = 0))
        label = int(label_support[i])
        data = data.permute(0, 2, 1).to(device)
        data = torch.cat((data, data))
        with torch.no_grad():
            feat = model.get_emb(data)[0, :]
        feat = feat.detach().cpu().numpy().tolist()
        feats_train.append(feat)
        labels_train.append(label)

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)

    feats_test = []
    labels_test = []

    for i in range(k * n_q):
        data = torch.from_numpy(np.expand_dims(data_query[i], axis = 0))
        label = int(label_query[i])
        data = data.permute(0, 2, 1).to(device)
        data = torch.cat((data, data))
        with torch.no_grad():
            feat = model.get_emb(data)[0, :]
        feat = feat.detach().cpu().numpy().tolist()
        feats_test.append(feat)
        labels_test.append(label)

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
    
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats_train)
    model_tl = SVC(kernel ='linear')
    model_tl.fit(scaled, labels_train)
    # model_tl.fit(feats_train, labels_train)
    
    test_scaled = scaler.transform(feats_test)
    
    # accuracy = model_tl.score(feats_test, labels_test) * 100
    accuracy = model_tl.score(test_scaled, labels_test) * 100
    acc.append(accuracy)

    # print(f"C = {c} : {model_tl.score(test_scaled, labels_test)}")
    # print(f"Run - {run + 1} : {accuracy}")
    
print(f'{np.mean(acc)} +/- {np.std(acc)}')