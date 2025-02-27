import argparse
import random
import numpy as np
from sklearn.svm import SVC
import os
import torch
from torch.utils.data import DataLoader

from data import ModelNet40SVM, ScanObjectNNSVM
from model import DCP
from model_mah import DCP as MDCP



parser = argparse.ArgumentParser(description='Point Cloud Recognition')
parser.add_argument('--num_points', type=int, default=1024,
                    help='num of points to use')
parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40', 'scanobjectnn'],
                        help='Dataset to evaluate')
parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
parser.add_argument('--emb_nn', type=str, default='dgcnn', metavar='N',
                    choices=['pointnet', 'dgcnn'],
                    help='Embedding nn to use, [pointnet, dgcnn]')
parser.add_argument('--pointer', type=str, default='identity', metavar='N',
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


parser.add_argument('--model_path', type=str, default='/home/phoenix/Experiments/Tejas/dcp/Checkpoints/dcp_v1/models/model.best.t7', metavar='N',
                        help='Pretrained model path')
parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp', 'mdcp'],
                        help='Dataset to model')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')

args = parser.parse_args()

device = torch.device("cuda")


torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.cuda.manual_seed(args.seed)
torch.set_printoptions(10)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(args.seed)
random.seed(args.seed)

#Try to load models

if args.model.lower() == "dcp":
    model = DCP(args).to(device)
elif args.model.lower() == "mdcp":
    model = MDCP(args).to(device)


# model = torch.nn.DataParallel(model).to(device)
state_dict = torch.load(args.model_path,map_location='cuda')
model.load_state_dict(state_dict)


# mstate_dict = torch.load(args.model_path_m,map_location='cuda')
# mmodel.load_state_dict(mstate_dict)
# new_state_dict = {}
# for key in state_dict:
#   new_key = key.replace('module.','')
#   new_state_dict[new_key] = state_dict[key]

# model.load_state_dict(new_state_dict)

print("Model Loaded !!")




if args.dataset.lower() == "modelnet40":
    train_loader = DataLoader(ModelNet40SVM(partition='train', num_points=args.num_points),
                                batch_size=32, shuffle=True)
    test_loader = DataLoader(ModelNet40SVM(partition='test', num_points=args.num_points),
                                batch_size=16, shuffle=False)
    print('MN40 Done !!')

elif args.dataset.lower() == "scanobjectnn":

    train_loader = DataLoader(ScanObjectNNSVM(partition='train', num_points=args.num_points),
                                batch_size=32, shuffle=True)
    test_loader = DataLoader(ScanObjectNNSVM(partition='test', num_points=args.num_points),
                                batch_size=16, shuffle=False)
    print('SONN Done !!')






feats_train = []
labels_train = []
model = model.eval()


for i, (data, label) in enumerate(train_loader):
    if args.dataset.lower() == "modelnet40":
        labels = list(map(lambda x: x[0],label.numpy().tolist()))
    elif args.dataset.lower() == "scanobjectnn":
        labels = label.numpy().tolist()
    data = data.permute(0, 2, 1).to(device)
    with torch.no_grad():
        feats = model.get_emb(data)
    feats = feats.cpu().numpy()

    for feat in feats:
        feats_train.append(feat)
    labels_train += labels

feats_train = np.array(feats_train)

labels_train = np.array(labels_train)




feats_test = []

labels_test = []
model = model.eval()

for i, (data, label) in enumerate(test_loader):
    if args.dataset.lower() == "modelnet40":
        labels = list(map(lambda x: x[0],label.numpy().tolist()))
    elif args.dataset.lower() == "scanobjectnn":
        labels = label.numpy().tolist()
    data = data.permute(0, 2, 1).to(device)
    with torch.no_grad():
        feats = model.get_emb(data)
    feats = feats.cpu().numpy()

    for feat in feats:
        feats_test.append(feat)

    labels_test += labels

feats_test = np.array(feats_test)

labels_test = np.array(labels_test)



# c = 0.1 # Linear SVM parameter C, can be tuned
print(args.dataset, args.model, args.pointer)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler


mmodel_tl = SVC(kernel ='linear')




scaler = StandardScaler()
scaled = scaler.fit_transform(feats_train)
model_tl = SVC(kernel ='linear')
model_tl.fit(scaled, labels_train)
test_scaled = scaler.transform(feats_test)
print(f"{model_tl.score(test_scaled, labels_test)}")



# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# dcp_emb = TSNE(n_components=2,learning_rate='auto',init='random', perplexity=30, random_state=args.seed).fit_transform(feats_test)
# plt.scatter(dcp_emb[:,0],dcp_emb[:,1],s=1,c=labels_test,cmap="Spectral")
# plt.savefig(f"./plots/dcp_{args.pointer}_{args.dataset}_emb.png")
# mdcp_emb = TSNE(n_components=2,learning_rate='auto',init='random', perplexity=30, random_state=args.seed).fit_transform(mfeats_test)
# plt.clf()
# plt.scatter(mdcp_emb[:,0],mdcp_emb[:,1],s=1,c=labels_test,cmap="Spectral")
# plt.savefig(f"./plots/mdcp_{args.pointer}_{args.dataset}_emb.png")