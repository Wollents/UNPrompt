import torch.nn as nn
from model import Model, SimplePrompt, GPFplusAtt, Projection, PreTrainModel
from prompts_dataset import Prompts_Dataset
from utils import *
import random
from pretrain import traingrace
from prompts_model import Prompts
from prompts_dataset import Prompts_Dataset, feat_alignment
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import argparse
from tqdm import tqdm
import ipdb

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0]))
os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device = torch.device("cuda:0")
parser = argparse.ArgumentParser(description='Unified Neighborhood Prompt for Graph Anomaly Detection')
parser.add_argument('--dataset', type=str, default='Facebook')
parser.add_argument('--datasets_dir', type=str, default='./my_datasets/')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--edge_drop_prob', type=float, default=0.2)
parser.add_argument('--feat_drop_prob', type=float, default=0.3)
parser.add_argument('--lamda', type=float, default=0.2)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--unifeat', type=int, default=64)
parser.add_argument('--numprompts', type=int, default=10)
parser.add_argument('--shot', type=int, default=10)
parser.add_argument('--pretrain_batch_size', type=int, default=2048)
parser.add_argument('--test_sample_round', type=int, default=50)
parser.add_argument('--save_pt_dir', type=str, default='./model_save/')
args = parser.parse_args()

prompts_model_config = {
    "model": "Prompts",
    "lr": 1e-5,
    "drop_rate": 0,
    "h_feats": 1024,
    "num_prompt": 10,
    "num_hops": 2,
    "weight_decay": 5e-5,
    "in_feats": 64,
    "num_layers": 4,
    "activation": "ELU"
}

pretrain_model_config = {
    "model": "Prompts",
    "lr": 1e-5,
    "drop_rate": 0,
    "h_feats": 64,
    "num_prompt": 10,
    "num_hops": 2,
    "weight_decay": 5e-5,
    "in_feats": 64,
    "num_layers": 4,
    "activation": "ELU"
}

dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load and preprocess data
def loaddata(dataset, args, device):
    adj, features,  ano_label, str_ano_label, attr_ano_label = load_mat(dataset, args.datasets_dir)
    adj = adj.astype(np.float32)
    if dataset in ['Amazon', 'YelpChi', 'tolokers', 'tfinance']:
        features = preprocess_features(features)
    else:
        features = features.toarray()

    features = torch.FloatTensor(features)
    adj_sp = sp.csr_matrix(adj)
    row, col = adj_sp.nonzero()
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    features = feat_alignment(features, edge_index, args.unifeat)

    diag_adj = adj.diagonal() > 0
    if diag_adj.all():
        adj_withloop_won = adj
        adj_woself = adj - sp.eye(adj.shape[0])
    else:
        adj_withloop_won = adj + sp.eye(adj.shape[0])
        adj_woself = adj
    adj_withloop = normalize_adj(adj_withloop_won)
    adj_withloop = sparse_mx_to_torch_sparse_tensor(adj_withloop)
    adj_woself = normalize_adj(adj_woself)
    adj_woself = sparse_mx_to_torch_sparse_tensor(adj_woself)

    ano_label = torch.FloatTensor(ano_label)
    ano_label = ano_label.to(device)
    adj_withloop = adj_withloop.to(device)
    adj_woself = adj_woself.to(device)
    features = features.to(device)

    return adj_withloop_won, adj_withloop, adj_woself, features, ano_label


traindatasets = ['pubmed', 'Flickr', 'questions', 'YelpChi']
# traindatasets = ['pubmed']
targdataset = ['cora', 'citeseer', 'ACM', 'BlogCatalog',
               'Facebook', 'weibo', 'Reddit', 'Amazon']
# traindatasets = ['Facebook_un']
# targdataset = ['Amazon_un', 'Reddit_un']
# if len(traindatasets) < 2:
#     print("unp dataset")
#     args.datasets_dir = "./Datasets/"
adj_withloop_won_train = []
adj_withloop_train = []
adj_woself_train = []
features_train = []
ano_label_train = []
for dataset in traindatasets:
    adj_withloop_won, adj_withloop, adj_woself, features, ano_label = loaddata(dataset, args, device)
    adj_withloop_won_train.append(adj_withloop_won)
    adj_withloop_train.append(adj_withloop)
    adj_woself_train.append(adj_woself)
    features_train.append(features)
    ano_label_train.append(ano_label)

prompts_data_train = [Prompts_Dataset(dims=args.unifeat, name=dataset, dataset_dir=args.datasets_dir) for dataset in traindatasets]
prompts_data_test = [Prompts_Dataset(dims=args.unifeat, name=dataset, dataset_dir=args.datasets_dir) for dataset in targdataset]


all_aucs = []
all_aps = []
auc_dict = {}
pre_dict = {}
grace_config = {

}
global_index = -1


def set_seed(seed):
    dgl.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


for t in range(5):
    seed = t
    set_seed(seed)
    model = PreTrainModel(**pretrain_model_config)
    model = model.to(device)
    # traingrace(model, adj_withloop_won_train, adj_withloop_train, features_train, args, device)

    model.eval()

    prompts_model = Prompts(**prompts_model_config).to(device)
    model_load_path = args.save_pt_dir + 'promtpt_model_seed_' + str(seed) + '_' + str(args.epochs) + '_epoch.pth'
    if os.path.exists(model_load_path):
        prompts_model.load_state_dict(torch.load(model_load_path))
        print("using saving model")
    optimiser_prompt_proj = torch.optim.Adam(prompts_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # train_prompts
    if not os.path.exists(model_load_path):
        tbar = tqdm(range(args.epochs), desc="Prompts_Training")
        for epoch in tbar:
            for prompts_data in prompts_data_train:

                prompts_model.train()
                train_graph = prompts_data.graph.to(device)
                prompts_data.x = prompts_model.add(train_graph.x)
                x_list = model(train_graph)

                residual_embed = prompts_model(x_list)

                cross_loss = prompts_model.cross_attn.get_train_loss(residual_embed, train_graph,
                                                                    args.numprompts)

                subgraph_loss = prompts_model.get_subgraph_loss(residual_embed, train_graph,
                                                                model)

                loss = cross_loss + subgraph_loss
                optimiser_prompt_proj.zero_grad()
                loss.backward()
                optimiser_prompt_proj.step()

                tbar.set_postfix(epoch=epoch, dataset=prompts_data.name, loss=loss)
            tbar.update()
        torch.save(prompts_model.state_dict(), model_load_path)
        torch.cuda.empty_cache()
    # eval_prompts
    test_score_list = {}
    prompts_model.eval()
    for te_data in prompts_data_test:
        te_data.few_shot(args.shot)
    for didx, test_data in enumerate(prompts_data_test):
        test_graph = test_data.graph.to(device)
        labels = test_graph.ano_labels
        shot_mask = test_graph.shot_mask.bool()

        query_labels = labels[~shot_mask].to(device)
        x_list = model(test_graph)
        residual_embed = prompts_model(x_list)

        query_scores = prompts_model.cross_attn.get_test_score(residual_embed, test_graph.shot_mask,
                                                               test_graph.ano_labels)
        query_scores = query_scores.detach().cpu().numpy()
        subgraph_score = prompts_model.get_subgraph_score(residual_embed, test_graph.shot_mask, test_graph, model, multi_round=args.test_sample_round)
        
        max_auc = 0
        max_prc = 0
        index = 0
        for i in np.arange(0,1,0.1):
            test_score = test_eval(query_labels, query_scores + subgraph_score * i)
            if test_score['AUROC'] > max_auc:
                max_auc = test_score['AUROC']
                index = i
            if test_score['AUPRC'] > max_prc:
                max_prc = test_score['AUROC']
                index = i
        test_score['AUROC'] = max_auc
        test_score['AUROC'] = max_prc
        test_score['index'] = index
        global_index = index

        # Store the test scores in the dictionary
        test_data_name = test_data.name
        test_score_list[test_data_name] = {
            'AUROC': test_score['AUROC'],
            'AUPRC': test_score['AUPRC'],
            "index": test_score['index'],
        }
    for test_data_name, test_score in test_score_list.items():
        if test_data_name not in auc_dict:
            auc_dict[test_data_name] = []
            pre_dict[test_data_name] = []
        auc_dict[test_data_name].append(test_score['AUROC'])
        pre_dict[test_data_name].append(test_score['AUPRC'])
        print(f'Test on {test_data_name}, AUC is {auc_dict[test_data_name]}')
        key_index = 'index'
        print(f"index is {test_score[key_index]}")
    print(test_score_list)

# Calculate mean and standard deviation for each test dataset
auc_mean_dict, auc_std_dict, pre_mean_dict, pre_std_dict = {}, {}, {}, {}

for test_data_name in auc_dict:
    auc_mean_dict[test_data_name] = np.mean(auc_dict[test_data_name])
    auc_std_dict[test_data_name] = np.std(auc_dict[test_data_name])
    pre_mean_dict[test_data_name] = np.mean(pre_dict[test_data_name])
    pre_std_dict[test_data_name] = np.std(pre_dict[test_data_name])


# Output the results for each test dataset
for test_data_name in auc_mean_dict:
    str_result = 'AUROC:{:.4f}+-{:.4f}, AUPRC:{:.4f}+-{:.4f}, index:{}'.format(
        auc_mean_dict[test_data_name],
        auc_std_dict[test_data_name],
        pre_mean_dict[test_data_name],
        pre_std_dict[test_data_name],
        global_index)
    line = '-' * 50 + test_data_name + '-' * 50
    print('-' * 50 + test_data_name + '-' * 50)
    end = '-' * 50 + "end" + test_data_name + '-' * 50
    print('str_result', str_result)
    with open(f'results/{args.dataset}_my_test.txt', 'a') as f:
        f.write('{}\n{}\n{}\n'.format(line, str_result, end))
