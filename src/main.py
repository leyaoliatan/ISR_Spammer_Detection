#src/main.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import get_laplacian
import scipy.sparse as sp

from time import perf_counter
import datetime
import os
import pandas as pd
import networkx as nx
from community import community_louvain
from time import perf_counter



# active learning unlabeled data selection strategies
from active_learning.strategies import *

from utils.config_utils import load_config, get_device
from utils.utils import augment_feature, print_graph_info, sgc_precompute, set_seed, get_classes_statistic, plot
from models.models import get_model
from utils.metrics import accuracy, f1, f1_isr 
#f1_isr added a small volume to avoid division by zero considering the imbalanced data
#f1 gives both micro and macro f1 scores, we focus on macro f1 score


# from models.gcn import GCN  # 假设您的GCN模型在这个位置
# from utils.data_utils import load_data, preprocess_features
# from utils.train_utils import train_GCN, test_GCN

from utils.args import get_citation_args
args = get_citation_args()#get all the arguments
print("args.train_size",args.train_size)


def ensure_nonrepeat(idx_train, selected_nodes):
    for node in idx_train:
        if node in selected_nodes:
            raise Exception(
                'In this iteration, the node {} need to be labelled is already in selected_nodes'.format(node))
    return


##################

def loss_function_laplacian_regularization(output, train_labels, edge_index):
    loss_cls = F.cross_entropy(output, train_labels)
    
    # 计算稀疏格式的图拉普拉斯矩阵
    lap_sp = get_laplacian(edge_index, normalization='sym')[0]
    lap_sp = sp.FloatTensor(lap_sp)
    
    # 使用稀疏矩阵乘法计算损失
    loss_lap = torch.sum((lap_sp @ output.T) ** 2)
    
    return loss_cls + 0.1 * loss_lap

def loss_function_consistency_regularization(model, x, edge_index, train_labels, selected_nodes):
    # 计算原始输入的预测结果
    output = model(x, edge_index)
    output_select = output[selected_nodes, :]
    loss_cls = F.cross_entropy(output_select, train_labels)
    
    # Generate adversarial samples by adding random noise
    adv_x = x + 0.1 * torch.randn_like(x)

    # Compute the adversarial output
    adv_output = model(adv_x, edge_index)

    # Compute the consistency loss
    loss_cons = F.kl_div(adv_output, output.detach(), reduction='batchmean')

    return loss_cls + 0.1 * loss_cons

def loss_function_subgraph_regularization(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    adj = edge_index.coalesce()
    
    # 将稀疏邻接矩阵转换为稠密矩阵
    adj = adj.to_dense()
    
    # 计算节点隶属度
    node_membership = model.get_node_embedding()
    # node_membership = node_membership.T
    
    # 计算同一子图内节点预测差异的平方和
    loss_subgraph = torch.sum(torch.matmul(node_membership.T, torch.matmul(adj, node_membership)))
    
    return loss_cls + 0.1 * loss_subgraph

def loss_function_subgraph_regularization_v1(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    adj = edge_index.coalesce()
    
    # 将稀疏邻接矩阵转换为稠密矩阵
    adj = adj.to_dense()
    
    # 计算节点隶属度
    node_membership = model.get_node_embedding()

    # 计算邻域内节点预测相似性
    # 计算邻域内节点预测的相似度矩阵
    sim_matrix = torch.matmul(output, output.T)  # 计算预测值的相似度矩阵
    neighborhood_sim = torch.sum(sim_matrix * adj)  # 在邻接矩阵中加权相似度矩阵
    
    # 最小化相似性
    loss_similarity = torch.mean(neighborhood_sim)
    
    return loss_cls + 0.1 * loss_similarity

def loss_function_local_consistency(model, x, edge_index, train_labels, selected_nodes):
    output = model(x, edge_index)
    output_selected = output[selected_nodes, :]
    
    # 分类损失
    loss_cls = F.cross_entropy(output_selected, train_labels)
    
    # 计算邻接矩阵
    adj = edge_index.coalesce()
    adj = adj.to_dense()
    
    # 获取节点嵌入
    node_membership = model.get_node_embedding()
    
    # 计算每个节点及其邻域的预测差异
    loss_local_consistency = 0
    for node in selected_nodes:
        neighbors = torch.nonzero(adj[node, :]).squeeze()  # 获取邻域节点
        if len(neighbors) > 0:
            node_output = output[node, :]
            neighbors_output = output[neighbors, :]
            # 计算预测差异
            loss_local_consistency += torch.sum((node_output - neighbors_output) ** 2)
    
    return loss_cls + 0.1 * loss_local_consistency



def train_GCN(model, adj, selected_nodes, val_nodes,
             features, train_labels, val_labels,
             epochs=args.epochs, weight_decay=args.weight_decay,
             lr=args.lr, dropout=args.dropout):
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    best_acc_val = 0
    should_stop = False
    stopping_step = 0

    # preconditioner = psgd.KFAC(
    #     model, 
    #     eps=0.01, 
    #     sua=False, 
    #     pi=False, 
    #     update_freq=50,
    #     alpha=1.,
    #     constraint_norm=False
    # )
    gamma = 2.0
    for epoch in range(epochs):#by default 300
        # lam = (float(epoch)/float(epochs))**gamma if gamma is not None else 0.
        # model.train()
        # optimizer.zero_grad()
        # output = model(features, adj)
        # label = output.max(1)[1]
        # label[selected_nodes] = train_labels
        # label.requires_grad = False
        
        # loss = F.nll_loss(output[selected_nodes], label[selected_nodes])
        
        # training_mask = torch.ones(output.size(0), dtype=torch.bool)
        # training_mask[selected_nodes] = False
        # loss += lam * F.nll_loss(output[training_mask], label[training_mask])
        
        # loss.backward(retain_graph=True)
        # if preconditioner:
        #     preconditioner.step(lam=lam)
        # optimizer.step()

        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index=adj)
        output = output[selected_nodes, :] #train set
        # print(f'output.size(): {output.size()}')

        # loss_train = F.cross_entropy(output, train_labels)
        loss_train = F.nll_loss(output, train_labels)
        # loss_train = loss_function_laplacian_regularization(output, train_labels, adj)
        # loss_train = loss_function_consistency_regularization(model, features, adj, train_labels, selected_nodes)
        loss_train = loss_function_subgraph_regularization(model, features, adj, train_labels, selected_nodes)
        # loss_train = loss_function_local_consistency(model, features, adj, train_labels, selected_nodes)

        # loss_train.backward()
        loss_train.backward(retain_graph=True)
        optimizer.step()
        

    train_time = perf_counter() - t

    with torch.no_grad():
        model.eval()
        output = model(features, adj)
        output = output[val_nodes, :] #validation set, initial is 10
        acc_val = accuracy(output, val_labels)
        micro_val, macro_val, micro_recall, macro_recall, micro_precision, macro_precision = f1(output, val_labels) #f1 score on the validation set
        print('acc_val: {}'.format(acc_val))
        f1_val, recall_val, precision_val = f1_isr(output, val_labels) # performance on the validation set
        print('f1_val_isr: {}'.format(f1_val))
    return model, acc_val, micro_val, macro_val, micro_recall, macro_recall, micro_precision, macro_precision, train_time, f1_val, recall_val, precision_val

def test_GCN(model, adj, test_mask, features, test_labels, all_test_labels, all_test_idx, save = args.save_pred, save_path = args.save_path, save_name=f"{args.model}_{args.strategy}_preds", version = args.version_name):
    model.eval()
    with torch.no_grad():
        output_all = model(features, adj)
        if save: #save prediction results on all test set:
            if all_test_idx is not None:
                print("---> Saving Predictions --->")
                output_test = output_all[all_test_idx, :] #prediction on all unsed data points
                output_test_preds = output_test.max(1)[1] #predicted label
                output_test_probs = torch.softmax(output_test, dim=1)
                max_probs = output_test_probs.max(1)[0]  # 最大概率值

                # DataFrame
                df = pd.DataFrame({
                    'node_id': all_test_idx,
                    'predicted_label': output_test_preds.cpu().numpy(),
                    'predicted_prob': max_probs.cpu().numpy(),
                    })
                
                # to CSV
                csv_path = f"{save_path}/{save_name}_{version}.csv"
                df.to_csv(csv_path, index=False)
                print(f"---- Predictions saved ---->>>")

    output = output_all[test_mask, :]
    acc_test = accuracy(output, test_labels)
    micro_test, macro_test, micro_recall, macro_recall, micro_precision, macro_precision = f1(output, test_labels)

    f1_test, recall_test, precision_test = f1_isr(output, test_labels)

    output_test = output_all[all_test_idx, :]
    f1_test_all, recall_test_all, precision_test_all = f1_isr(output_test, all_test_labels)
    print(f'f1_test: {f1_test}, recall_test: {recall_test}, precision_test: {precision_test}')
    print(f'f1_test_all: {f1_test_all}, recall_test_all: {recall_test_all}, precision_test_all: {precision_test_all}')

    # return acc_test, micro_test, macro_test, f1_test, recall_test, precision_test
    return acc_test, micro_test, macro_test, micro_recall, macro_recall, micro_precision, macro_precision, f1_test_all, recall_test_all, precision_test_all

#*Note: isr adjusted performance is the micro avg (adjusted in the sense of avoiding division by zero) over all untouched nodes


##################

class run_wrapper():
    def __init__(self, config, dataset):
        self.config = config
        self.device = get_device(config)
        self.model = args.model
        self.strategy = args.strategy
        self.degree = args.degree

        if dataset in ['spammer','amazon','yelp']:
            self.graph = None
            #load data path
            network_path = os.path.join(config['data']['path'], config['data']['network_file'])
            features_path = os.path.join(config['data']['path'], config['data']['feature_file'])
            labels_path = os.path.join(config['data']['path'], config['data']['label_file'])

            ## load network
            print("start loading J01Network")
            graph_data = np.loadtxt(network_path, delimiter=' ', dtype=int) #graph_data is a numpy array
            graph_data[:,0] = graph_data[:,0] - 1
            graph_data[:,1] = graph_data[:,1] - 1 # 0-based index, the stored txt is a 1-based index edge list
            self.nx_G = nx.Graph() # create a new graph
            self.nx_G.add_edges_from(graph_data) # add edges to the graph
            #suumary of the network
            print("J01Network summary:")
            print_graph_info(self.nx_G)

            #construct the adjacency matrix
            edge_tensor = torch.from_numpy(graph_data).long() #input is a numpy array of edges, this step transfer the numpy array to tensor
            indices = edge_tensor.t().contiguous() #transpose the edge_tensor and make it contiguous, this is to make the edge_tensor to be a 2D tensor where each column is an edge, we then use this tensor to construct the sparse tensor
            num_edges = edge_tensor.shape[0] #get the indices of the edges
            values = torch.ones(num_edges) #we assign a value of 1 to each edge
            num_nodes = edge_tensor.max().item() + 1
            adj = torch.sparse_coo_tensor(indices, values, size=(num_nodes, num_nodes)) #the inputs are the non-zero elements (indices and values) of the sparse tensor, the size of the sparse tensor
            adj = adj.coalesce()
            if torch.cuda.is_available():
                adj = adj.to('cuda:0')
                adj = adj.cuda() #transfer the sparse tensor to cuda
            row_sum = torch.sparse.sum(adj, dim=1).to_dense()#this returns the degree of each node
            row_sum[row_sum == 0] = 1  # 避免除以零
            values_normalized = 1.0 / row_sum[adj.indices()[0]]
            adj_normalized = torch.sparse_coo_tensor(adj.indices(), values_normalized, adj.size())#normalize the adjacency matrix, this balanaize the importance of nodes with different degrees
            self.adj = adj_normalized


            #load features
            print("start loading features")
            features = np.loadtxt(features_path, delimiter='\t')#np array of features
            features = augment_feature(features, self.nx_G) #add k-core and modularity as features
            if torch.cuda.is_available():
                self.features = torch.from_numpy(features).float().cuda()
            else:
                self.features = torch.from_numpy(features).float()

            #load labels
            print("start loading labels")
            labels_data = pd.read_csv(labels_path, delimiter=' ', usecols = [1,2])
            labels_data = labels_data.to_numpy()
            if torch.cuda.is_available():
                self.labels = torch.from_numpy(labels_data[:, 1]).cuda()
            else:
                self.labels = torch.from_numpy(labels_data[:, 1])
            
            #load train and test indices --> use 50% labeled data to train and 50% to test
            print("start loading labeled indices")
            traning_data_path = os.path.join(config['data']['path'], config['data']['train_file'])
            training_data = np.loadtxt(traning_data_path, delimiter=' ', dtype=int)
            testing_data_path = os.path.join(config['data']['path'], config['data']['test_file'])
            testing_data = np.loadtxt(testing_data_path, delimiter=' ', dtype=int)
            if torch.cuda.is_available():
                self.idx_test = torch.from_numpy(testing_data[:,0] - 1).cuda()
            else:
                self.idx_test = torch.from_numpy(testing_data[:,0] - 1)
            self.idx_non_test = (training_data[:,0]-1).tolist() #original index is 1-based
        
        #so far, we have loaded the data and constructed the graph
        self.dataset = dataset
        print(f'self.labels: {self.labels, self.labels.shape}')
        print(f'self.adj: {self.adj}')
        print(f'self.feature: {self.features, self.features.shape}')
        print(f'self.idx_test is {len(self.idx_test)}, self.idx_non_test is {len(self.idx_non_test)}')
        print('Initial train indices:', self.idx_non_test[:5],'...')
        print('Initial test indices:', self.idx_test[:5],'...')
        print('finished loading dataset')
        print('-----------------------------------------------------------')
        self.raw_features = self.features
        
        if self.model == "SGC": #simple graph convolution
            self.features, precompute_time = sgc_precompute(self.features, self.adj, self.degree)
            print("{:.4f}s".format(precompute_time))
            if self.strategy == 'featprop':
                self.dis_features = self.features
        else:
            if self.strategy == 'featprop':
                self.dis_features, precompute_time = sgc_precompute(self.features, self.adj, self.degree)

    def run(self, strategy, num_labeled_list=[10,15,20,25,30,35,40,50],max_budget=160,seed=1): #Q1: why 160? why set the budget list as this?
        set_seed(seed, args.cuda)
        budget = num_labeled_list[-1] #set the max_budget as the last element of the num_labeled_list
        if strategy in ['ppr', 'pagerank', 'pr_ppr', 'mixed', 'mixed_random', 'unified']:
            print('strategy is ppr or pagerank')
            # nx_G = nx.from_dict_of_lists(self.graph)
            nx_G = self.nx_G
            PR_scores = nx.pagerank(nx_G, alpha=0.85)
            # print('PR_scores: ', PR_scores)
            nx_nodes = nx.nodes(nx_G)
            original_weights = {}
            for node in nx_nodes:
                original_weights[node] = 0.
        
        idx_non_test = self.idx_non_test.copy() #tranning data indices
        print('len(idx_non_test) is {}'.format(len(idx_non_test)))
        #validation set
        num_val = 10 #validation set size
        idx_val = np.random.choice(idx_non_test, num_val, replace=False) #choose 10 indices from the training data
        idx_non_test = list(set(set(idx_non_test) - set(idx_val))) #remove the validation set from the training data

        #initially select some nodes
        L = 5 #initially select 5 nodes #Q2: why set the initial number of nodes as 5?
        selected_nodes = np.random.choice(idx_non_test, L, replace=False) #initialize with 5 nodes
        print(f'--> Initalize with {L} selected_nodes:', selected_nodes)
        idx_non_test = list(set(set(idx_non_test) - set(selected_nodes)))

        model = get_model(args.model, 
                          nfeat = self.features.size(1), 
                          nclass=2, 
                          nhid=args.hidden, 
                          dropout = args.dropout,
                          cuda = args.cuda) #model is assigned by the model we choose in the arguments, the current one is GIN-adv
        
        budget = 20 #Q3: why set the budget as 20?
        pool = idx_non_test #initial pool is the training data
        print(f'len(idx_non_test) after dropping {L} initial node and {num_val} validation node: {len(idx_non_test)}')
        np.random.seed() # cancel the fixed seed
        all_test_idx = list(set(set(self.idx_test.cpu().numpy().tolist()).union(set(pool)))) #combine the test indices and the pool indices
        # 验证 all_test_idx 中没有重复节点
        assert len(all_test_idx) == len(set(all_test_idx)), "all_test_idx contains duplicate nodes!"
        pool = list(set(set(self.idx_test.cpu().numpy().tolist()).union(set(pool)))) #same for pool, we include the test data in the pool for future selecting

        if args.model == 'GIN_adv': #we are actually using GIN-adv now
            args.lr = 0.01
            model, acc_val, micro_val, macro_val, micro_recall, macro_recall, micro_precision, macro_precision, train_time, f1_val, recall_val, precision_val = train_GCN(model, self.adj, selected_nodes, idx_val, self.features,
                                                                                self.labels[selected_nodes],
                                                                                self.labels[idx_val],
                                                                                args.epochs, args.weight_decay, args.lr,
                                                                                args.dropout)#initially, we use 5 nodes to train the model
        print('-------------initial results------------')
        #test, train, val size:
        print('train size:', len(selected_nodes))
        print('val size:', len(idx_val))
        print('test size:', len(self.idx_test))
        print(f'pool size: {len(pool)}')
        print(f'len(all_test_idx): {len(all_test_idx)}')
        print('On the validation set:...')
        print('acc_val: {:.4f}'.format(acc_val))
        print('f1: micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
        print('recall: micro_recall: {:.4f}, macro_recall: {:.4f}'.format(micro_recall, macro_recall))
        print('precision: micro_precision: {:.4f}, macro_precision: {:.4f}'.format(micro_precision, macro_precision))
        print('isr adjusted: f1_val: {:.4f}, recall_val: {:.4f}, precision_val: {:.4f}'.format(f1_val, recall_val, precision_val))
        print('----------------------------------------')

        # start active learning with selected strategy
        print('start active learning')
        print('strategy:', strategy)
        cur_num = 0 #track the selected number of nodes
        val_results = {'acc': [], 'micro': [], 'macro': [], 
                       'micro_recall': [], 'macro_recall': [],
                       'micro_precision': [], 'macro_precision': [],
                       'f1': [], "recall":[], "precision":[]}
        test_results = {'acc': [], 'micro': [], 'macro': [], 
                       'micro_recall': [], 'macro_recall': [],
                       'micro_precision': [], 'macro_precision': [],
                        'f1': [], "recall":[], "precision":[]}

        uncertainty_results = {}
        #initialize for random walk or unified strategy:
        if strategy == 'rw':
            self.walks = remove_nodes_from_walks(self.walks, selected_nodes)
        if strategy == 'unified':
            nodes = nx.nodes(nx_G)
            uncertainty_score = get_uncertainty_score(model, self.features, nodes)
            init_weights = {n: float(uncertainty_score[n]) for n in nodes}
            for node in selected_nodes:
                init_weights[node] = 0
            uncertainty_results[5] = {'selected_nodes': selected_nodes.tolist(), 'uncertainty_scores': init_weights}
        
        time_AL = 0 #track the time of active learning
        for i in range(len(num_labeled_list)):
            if num_labeled_list[i] > max_budget:
                break #stop the loop if the budget is larger than the max_budget, which is 160 by default
            budget = num_labeled_list[i] - cur_num #to meet total number of num_labeled_list[i] nodes, current budget is the difference between the current number of nodes and the target number of nodes
            cur_num = num_labeled_list[i] #update the current number of nodes
            t1 = perf_counter()
            if strategy == 'random':
                idx_train = query_random(budget, pool)
            elif strategy == 'uncertainty':
                if args.model == 'GIN_adv':
                    idx_train = query_uncertainty_GCN(model, self.adj, self.features, budget, pool)
                else:
                    idx_train = query_uncertainty(model, self.features, budget, pool)
            elif strategy == 'largest_degrees':
                if args.dataset not in ['cora', 'citeseer', 'pubmed']:
                    idx_train = query_largest_degree(self.graph, budget, pool)
                else:
                    idx_train = query_largest_degree(nx.from_dict_of_lists(self.graph), budget, pool)
            elif strategy == 'coreset_greedy':
                idx_train = query_coreset_greedy(self.features, list(selected_nodes), budget, pool)
            elif strategy == 'featprop':
                idx_train = query_featprop(self.dis_features, budget, pool)
            elif strategy == 'pagerank':
                idx_train = query_pr(PR_scores, budget, pool)
            else:
                raise NotImplementedError('cannot find the strategy {}'.format(strategy))
            
            time_AL += perf_counter() - t1
            assert len(idx_train) == budget #check if the number of selected nodes is equal to the budget, if not, raise an error
            ensure_nonrepeat(idx_train, selected_nodes) #check if the selected nodes are in the existing train nodes. if so, raise an error
            selected_nodes = np.append(selected_nodes, idx_train) #now, the new train nodes are the combination of the existing train nodes and the selected nodes
            pool = list(set(set(pool) - set(idx_train))) #remove the selected nodes from the pool
            all_test_idx = list(set(set(self.idx_test.cpu().numpy().tolist()).union(set(pool)))) #the new test indices, test set + pool
            assert len(all_test_idx) == len(set(all_test_idx)), "all_test_idx contains duplicate nodes!"
            if args.model == 'GIN_adv':
                model, acc_val, micro_val, macro_val, micro_recall, macro_recall, micro_precision, macro_precision, train_time, f1_val, recall_val, precision_val = train_GCN(model, self.adj, 
                                                                                                                                                                              selected_nodes, idx_val, self.features,
                                                                                                                                                                              self.labels[selected_nodes],
                                                                                                                                                                              self.labels[idx_val],
                                                                                                                                                                              args.epochs, args.weight_decay, args.lr,
                                                                                                                                                                              args.dropout)
                acc_test, micro_test, macro_test, micro_recall_test, macro_recall_test, micro_precision_test, macro_precision_test, f1_test, recall_test, precision_test = test_GCN(model, self.adj, self.idx_test, self.features, 
                                                                                                                                                                                    self.labels[self.idx_test], 
                                                                                                                                                                                    self.labels[all_test_idx], 
                                                                                                                                                                                    all_test_idx) #self.idx_test is used for test mask to get the evaluation results
            print('-----------------')
            print(f"Now on the train size of {num_labeled_list[i]} nodes -->")
            print(f"Budget: {budget}")
            print(f"Size pool {len(pool)}")
            print('Current train size:', len(selected_nodes))
            print('val size:', len(idx_val))
            print('test size:', len(self.idx_test))
            print(f'all test size: {len(all_test_idx)}')
            print('On the validation set:...')
            print('acc_val: {:.4f}'.format(acc_val))
            print('f1: micro_val: {:.4f}, macro_val: {:.4f}'.format(micro_val, macro_val))
            print('recall: micro_recall: {:.4f}, macro_recall: {:.4f}'.format(micro_recall, macro_recall))
            print('precision: micro_precision: {:.4f}, macro_precision: {:.4f}'.format(micro_precision, macro_precision))
            print('isr adjusted: f1_val: {:.4f}, recall_val: {:.4f}, precision_val: {:.4f}'.format(f1_val, recall_val, precision_val))
            print('On the test set:...')
            print('acc_test: {:.4f}'.format(acc_test))
            print('f1: micro_test: {:.4f}, macro_test: {:.4f}'.format(micro_test, macro_test))
            print('recall: micro_recall_test: {:.4f}, macro_recall_test: {:.4f}'.format(micro_recall_test, macro_recall_test))
            print('precision: micro_precision_test: {:.4f}, macro_precision_test: {:.4f}'.format(micro_precision_test, macro_precision_test))
            print('isr adjusted on the all test set: f1_test: {:.4f}, recall_test: {:.4f}, precision_test: {:.4f}'.format(f1_test, recall_test, precision_test))
            print('-----------------')

            val_results['acc'].append(acc_val)
            val_results['micro'].append(micro_val)
            val_results['macro'].append(macro_val)
            val_results['micro_recall'].append(micro_recall)
            val_results['macro_recall'].append(macro_recall)
            val_results['micro_precision'].append(micro_precision)
            val_results['macro_precision'].append(macro_precision)
            val_results['f1'].append(f1_val)
            val_results['recall'].append(recall_val)
            val_results['precision'].append(precision_val)

            test_results['acc'].append(acc_test)
            test_results['micro'].append(micro_test)
            test_results['macro'].append(macro_test)
            test_results['micro_recall'].append(micro_recall_test)
            test_results['macro_recall'].append(macro_recall_test)
            test_results['micro_precision'].append(micro_precision_test)
            test_results['macro_precision'].append(macro_precision_test)
            test_results['f1'].append(f1_test)
            test_results['recall'].append(recall_test)
            test_results['precision'].append(precision_test)

        print('time_AL:', time_AL)
        print('-----------------')
        return val_results, test_results, get_classes_statistic(self.labels[selected_nodes].cpu().numpy()), time_AL





        


        
####################
##test
# if __name__ == '__main__':
#     config = load_config()
#     dataset = 'amazon'
#     run = run_wrapper(config, dataset)
#     print("run.graph:", run.graph)
#     print("run.device:", run.device)
#     #run model
#     run.run(run.strategy, num_labeled_list=[50,100, 150, 200, 300, 400, 1000],max_budget=1000,seed=1)

# formal run
if __name__ == '__main__':
    config = load_config()
    dataset =   config['data']['dataset']
    version = args.version_name
    print(f"== Use {args.train_size} of Data to Train on {dataset}, Version: {version}")
    if dataset == 'amazon':
        total_node = 9424
        #num_labeled_list =[i for i in range(10, int(args.train_size*total_node), 100)] #test
        num_labeled_list =[i for i in range(10, int(args.train_size*total_node), 20)] # we select 10% of the labeled dataset into the training set by the selected active learning strategy
    elif dataset == 'yelp':
        pass #to be added 
    num_interval = len(num_labeled_list)
    val_results = {'micro': [[] for _ in range(num_interval)],
                   'macro': [[] for _ in range(num_interval)],
                   'acc': [[] for _ in range(num_interval)],
                   'micro_recall': [[] for _ in range(num_interval)],
                   'macro_recall': [[] for _ in range(num_interval)],
                   'micro_precision': [[] for _ in range(num_interval)],
                   'macro_precision': [[] for _ in range(num_interval)],
                   'f1': [[] for _ in range(num_interval)],
                   'recall': [[] for _ in range(num_interval)],
                   'precision': [[] for _ in range(num_interval)]}

    test_results = {'micro': [[] for _ in range(num_interval)],
                    'macro': [[] for _ in range(num_interval)],
                    'acc': [[] for _ in range(num_interval)],
                    'micro_recall': [[] for _ in range(num_interval)],
                    'macro_recall': [[] for _ in range(num_interval)],
                    'micro_precision': [[] for _ in range(num_interval)],
                    'macro_precision': [[] for _ in range(num_interval)],
                    'f1': [[] for _ in range(num_interval)],
                    'recall': [[] for _ in range(num_interval)],
                    'precision': [[] for _ in range(num_interval)]}
    
    #run multiple experiments with different seeds:
    if args.file_io: #if use fixed seeds preset
        input_file = './ISR/radom_seed.txt'
        with open(input_file,'r') as f:
            seeds = f.readline()
        seeds = list(map(int, seeds.split(' ')))# two runs with 100, 200
    else:
        seeds = [52, 574, 641, 934, 12] #5 runs
    seed_idx_map = {i: idx for idx, i in enumerate(seeds)}
    num_run = len(seeds)
    wrapper = run_wrapper(config, dataset)
    print(f"Will run {num_run} random experiments")
    print("run.graph:", wrapper.graph)
    print("run.device:", wrapper.device)

    total_AL_time = 0
    for i in range(len(seeds)):
        print(f'======================Start round {i}======================')
        print('current seed is {}'.format(seeds[i]))
        val_dict, test_dict, classes_dict, cur_AL_time = wrapper.run(args.strategy, 
                                                                     num_labeled_list=num_labeled_list,
                                                                     max_budget=1000,
                                                                     seed=seeds[i])

        for metric in ['micro', 'macro', 'acc',
                       'micro_recall','macro_recall',
                       'micro_precision','macro_precision', 
                       'f1', 'recall', 'precision']:
            
            for j in range(len(val_dict[metric])):
                val_results[metric][j].append(val_dict[metric][j])
                test_results[metric][j].append(test_dict[metric][j])

        total_AL_time += cur_AL_time
    
    print('======================Finished all rounds======================')

    #calculate avg & std performance over multiple runs:
    val_avg_results = {'micro': [0. for _ in range(num_interval)],
                       'macro': [0. for _ in range(num_interval)],
                       'acc': [0. for _ in range(num_interval)],
                       'micro_recall':[0. for _ in range(num_interval)],
                       'macro_recall':[0. for _ in range(num_interval)],
                       'micro_precision':[0. for _ in range(num_interval)],
                       'macro_precision':[0. for _ in range(num_interval)],
                       'f1': [0. for _ in range(num_interval)],
                       'recall': [0. for _ in range(num_interval)],
                       'precision': [0. for _ in range(num_interval)]}
    test_avg_results = {'micro': [0. for _ in range(num_interval)],
                    'macro': [0. for _ in range(num_interval)],
                    'acc': [0. for _ in range(num_interval)],
                       'micro_recall':[0. for _ in range(num_interval)],
                       'macro_recall':[0. for _ in range(num_interval)],
                       'micro_precision':[0. for _ in range(num_interval)],
                       'macro_precision':[0. for _ in range(num_interval)],
                    'f1': [0. for _ in range(num_interval)],
                    'recall': [0. for _ in range(num_interval)],
                    'precision': [0. for _ in range(num_interval)]}
    val_std_results = {'micro': [0. for _ in range(num_interval)],
                        'macro': [0. for _ in range(num_interval)],
                        'acc': [0. for _ in range(num_interval)],
                       'micro_recall':[0. for _ in range(num_interval)],
                       'macro_recall':[0. for _ in range(num_interval)],
                       'micro_precision':[0. for _ in range(num_interval)],
                       'macro_precision':[0. for _ in range(num_interval)],
                        'f1': [0. for _ in range(num_interval)],
                        'recall': [0. for _ in range(num_interval)],
                        'precision': [0. for _ in range(num_interval)]}
    test_std_results = {'micro': [0. for _ in range(num_interval)],
                        'macro': [0. for _ in range(num_interval)],
                        'acc': [0. for _ in range(num_interval)],
                       'micro_recall':[0. for _ in range(num_interval)],
                       'macro_recall':[0. for _ in range(num_interval)],
                       'micro_precision':[0. for _ in range(num_interval)],
                       'macro_precision':[0. for _ in range(num_interval)],
                        'f1': [0. for _ in range(num_interval)],
                        'recall': [0. for _ in range(num_interval)],
                        'precision': [0. for _ in range(num_interval)]}
    
    for metric in ['micro', 'macro', 'acc',
                        'micro_recall','macro_recall',
                        'micro_precision','macro_precision', 
                        'f1', 'recall', 'precision']:
        for j in range(len(val_results[metric])):
            val_avg_results[metric][j] = np.mean(val_results[metric][j])
            test_avg_results[metric][j] = np.mean(test_results[metric][j])
            val_std_results[metric][j] = np.std(val_results[metric][j])
            test_std_results[metric][j] = np.std(test_results[metric][j])

    if args.model == 'GIN_adv':
        save_path = args.save_path
        print(f"Results will be saved to {save_path}")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    file_path = os.path.join(save_path, f'log_{dataset}_{args.model}_active_learning_{args.strategy}.txt')
    with open(file_path, 'a') as f:
        f.write('---------datetime: %s-----------\n' % datetime.datetime.now())
        f.write(f'Budget list: {num_labeled_list}\n')
        f.write(f'learning rate: {args.lr}, epoch: {args.epochs}, weight decay: {args.weight_decay}, hidden: {args.hidden}\n')
        f.write(f'{num_run} round of runs using seed.txt\n')
        for metric in ['micro', 'macro', 'acc',
                        'micro_recall','macro_recall',
                        'micro_precision','macro_precision', 
                        'f1', 'recall', 'precision']:
            f.write("Test_{}_f1 {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_avg_results[metric])))
            f.write("Test_{}_std {}\n".format(metric, " ".join("{:.4f}".format(i) for i in test_std_results[metric])))

        f.write("Average AL_Time: {}s\n".format(total_AL_time / len(seeds)))

        #plot
        index = num_labeled_list #use 10% of labeled data to trian the model
        data_list = [test_avg_results['macro'],test_avg_results['macro_recall'],test_avg_results['macro_precision'],test_avg_results['f1']]
        labels = ['Macro_F1_test', 'Macro_Recall_test', 'Macro_Precision_test', 'F1_ISR_test_all']
        figure_name = f'{dataset}_{args.model}_active_learning_{args.strategy}_{num_run}_run_on_{args.train_size}_{version}'
        title = f'{dataset}: {args.model}_active_learning_{args.strategy}, {num_run} Rounds Avg Performance'
        xlabel = 'Training Nodes Used'
        ylabel = 'Performance'

        # 调用绘图函数
        plot(index, data_list, figure_name, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel, save_path=save_path)
        print(f"saved to'{save_path}/{figure_name}.png'")