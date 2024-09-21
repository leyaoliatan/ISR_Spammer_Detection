#Active learning strategies
import random
from heapq import nlargest, nsmallest

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances
from time import perf_counter
import torch_geometric.utils as tgu
from sklearn.metrics.pairwise import euclidean_distances

def get_uncertainty_score(model, features, nodes_idx):
    model.eval()
    output = model(features[nodes_idx])
    prob_output = F.softmax(output, dim=1).detach()
    # log_prob_output = torch.log(prob_output).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    # print('prob_output[0]: ', prob_output[0])
    # print('log_prob_output[0]: ', log_prob_output[0])
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)  # contain node idx
    entropy = entropy.cpu().numpy()
    total_score = {}
    for idx, node in enumerate(nodes_idx):
        total_score[node] = entropy[idx] #uncertainty score is the entropy of the node
    return total_score

def remove_nodes_from_walks(walks, nodes):
    """
    从随机游走序列中删除指定节点
    """
    print('len(walks): ', len(walks))
    new_walks = []
    # print('len(new_walks): ', len(new_walks))
    for idx, walk in enumerate(walks):
        remove_flag = False
        for node in nodes:
            if node in walk:
                remove_flag = True
                break
        if not remove_flag:
            new_walks.append(walk)
    return new_walks

def assign_points_to_clusters(medoids, distances):
    distances_to_medoids = distances[:,medoids]
    clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
    clusters[medoids] = medoids
    return clusters

def compute_new_medoid(cluster, distances):
    # mask = np.ones(distances.shape)
    # print(f'distance[10,10]: {distances[10,10]}')
    # t1 = perf_counter()
    # mask[np.ix_(cluster,cluster)] = 0.
    # print(f'np.ix_(cluster,cluster): {np.ix_(cluster,cluster)}')
    # print(f'mask: {mask}')
    # print('time creating mask: {}s'.format(perf_counter()-t1))
    # input('before')
    # cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
    # print(f'cluster_distances: {cluster_distances}')
    # t1 = perf_counter()
    # print('cluster_distances.shape: {}'.format(cluster_distances.shape))
    # costs = cluster_distances.sum(axis=1)
    # print(f'costs: {costs}')
    # print('time counting costs: {}s'.format(perf_counter()-t1))
    # print(f'medoid: {costs.argmin(axis=0, fill_value=10e9)}')
    # return costs.argmin(axis=0, fill_value=10e9)
    cluster_distances = distances[cluster,:][:,cluster]
    costs = cluster_distances.sum(axis=1)
    min_idx = costs.argmin(axis=0)
    # print(f'new_costs: {costs}')
    # print(f'new_medoid: {cluster[min_idx]}')
    return cluster[min_idx]

def k_medoids(distances, k=3):
    # From https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py

    m = distances.shape[0] # number of points

    # Pick k random medoids.
    print('k: {}'.format(k))
    # curr_medoids = np.array([-1]*k)
    # while not len(np.unique(curr_medoids)) == k:
    #     curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
    curr_medoids = np.arange(m)
    np.random.shuffle(curr_medoids)
    curr_medoids = curr_medoids[:k]
    old_medoids = np.array([-1]*k) # Doesn't matter what we initialize these to.
    new_medoids = np.array([-1]*k)

    # Until the medoids stop updating, do the following:
    num_iter = 0
    while not ((old_medoids == curr_medoids).all()):
        num_iter += 1
        # print('curr_medoids: ', curr_medoids)
        # print('old_medoids: ', old_medoids)
        # Assign each point to cluster with closest medoid.
        t1 = perf_counter()
        clusters = assign_points_to_clusters(curr_medoids, distances)
        # print(f'clusters: {clusters}')
        # print('time assign point ot clusters: {}s'.format(perf_counter() - t1))
        # Update cluster medoids to be lowest cost point.
        t1 = perf_counter()
        for idx, curr_medoid in enumerate(curr_medoids):
            # print(f'idx: {idx}')
            cluster = np.where(clusters == curr_medoid)[0]
            # cluster = np.asarray(clusters == curr_medoid)
            # print(f'curr_medoid: {curr_medoid}')
            # print(f'np.where(clusters == curr_medoid): {np.where(clusters == curr_medoid)}')
            # print(f'cluster: {cluster}')
            new_medoids[curr_medoids == curr_medoid] = compute_new_medoid(cluster, distances)
            del cluster
        # print('time update medoids: {}s'.format(perf_counter() - t1))
        old_medoids[:] = curr_medoids[:]
        curr_medoids[:] = new_medoids[:]
        if num_iter >= 50:
            print(f'Stop as reach {num_iter} iterations')
            break
    print('total num_iter is {}'.format(num_iter))
    print('-----------------------------')
    return clusters, curr_medoids

def query_random(number, nodes_idx):
    return np.random.choice(nodes_idx, size=number, replace=False)


def query_largest_degree(nx_graph, number, nodes_idx):
    """
    选择度最大的节点的策略

    优点：
    - 选择网络中最有影响力的节点
    - 可以快速传播信息到整个网络

    缺点：
    - 可能忽视网络中的重要但低度节点
    - 不考虑节点的特征信息

    参数：
    graph: 图结构（NetworkX图对象）
    budget (int): 要选择的节点数量
    pool (list): 可供选择的节点池

    返回：
    list: 选中的节点索引列表
    """

    degree_dict = nx_graph.degree(nodes_idx)
    idx_topk = nlargest(number, degree_dict, key=degree_dict.get)
    # print(idx_topk)
    return idx_topk


def query_uncertainty(model, features, number, nodes_idx):
    """
    基于GCN不确定性的选择策略

    优点：
    - 选择模型最不确定的样本，有助于提高决策边界
    - 通常比随机选择更有效

    缺点：
    - 可能过于关注噪声或异常值
    - 在某些情况下可能导致采样偏差

    参数：
    model: 训练好的GCN模型
    adj: 邻接矩阵
    features: 节点特征
    budget (int): 要选择的节点数量
    pool (list): 可供选择的节点池

    返回：
    list: 选中的节点索引列表
    """    
    model.eval()
    output = model(features[nodes_idx])#only use features in the model
    prob_output = F.softmax(output, dim=1).detach()
    # log_prob_output = torch.log(prob_output).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    # print('prob_output: ', prob_output)
    # print('log_prob_output: ', log_prob_output)
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)#entropy = -sum(p(x)log(p(x))), it's  a measure of uncertainty
    # print('entropy: ', entropy)
    indices = torch.topk(entropy, number, largest=True)[1] #choose the indiced of the largest number of the entropy
    # print('indices: ', list(indices.cpu().numpy()))
    indices = list(indices.cpu().numpy()) #convert the indices tensor to numpy array
    return np.array(nodes_idx)[indices]
    # return indices

def query_uncertainty_GCN(model, adj, features, number, nodes_idx):
    model.eval()
    # output = model(features[nodes_idx])
    output = model(features, adj) #use both features and adj into the model
    output = output[nodes_idx, :]
    prob_output = F.softmax(output, dim=1).detach()
    # log_prob_output = torch.log(prob_output).detach()
    log_prob_output = F.log_softmax(output, dim=1).detach()
    # print('prob_output: ', prob_output)
    # print('log_prob_output: ', log_prob_output)
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)
    # print('entropy: ', entropy)
    indices = torch.topk(entropy, number, largest=True)[1]
    # print('indices: ', list(indices.cpu().numpy()))
    indices = list(indices.cpu().numpy())
    return np.array(nodes_idx)[indices]

def query_random_uncertainty(model, features, number, nodes_idx):
    model.eval()
    output = model(features[nodes_idx])
    prob_output = F.softmax(output, dim=1).detach()
    log_prob_output = torch.log(prob_output).detach()
    entropy = -torch.sum(prob_output*log_prob_output, dim=1)
    indices = torch.topk(entropy, 3*number, largest=True)[1] #首先选择熵最大的 3 * number 个节点，然后从中随机选择 number 个节点
    indices = np.random.choice(indices.cpu().numpy(), size=number, replace=False)
    return np.array(nodes_idx)[indices]


def query_coreset_greedy(features, selected_nodes, number, nodes_idx):
    """
    基于核心集的贪婪选择策略

    优点：
    - 选择能代表整个数据集的最小子集 --> max distances
    - 有助于保持数据的多样性

    缺点：
    - 计算成本较高
    - 可能在某些特定问题上不如其他方法有效

    参数：
    features: 节点特征
    labeled_nodes (list): 已标记的节点列表
    budget (int): 要选择的节点数量
    pool (list): 可供选择的节点池

    返回：
    list: 选中的节点索引列表
    """
    features = features.cpu().numpy()
    # print('nodes_idx: ', nodes_idx)
    def get_min_dis(features, selected_nodes, nodes_idx):
        Y = features[selected_nodes]
        X = features[nodes_idx]
        dis = pairwise_distances(X, Y)
        # print('dis: ', dis)
        return np.min(dis, axis=1) #return the min distance of each node to the selected nodes, we want to select the unlabeled node with the max min distance

    new_batch = []
    for i in range(number):
        if selected_nodes == []:
            ind = np.random.choice(nodes_idx)
        else:
            min_dis = get_min_dis(features, selected_nodes, nodes_idx)
            # print('min_dis: ', min_dis)
            ind = np.argmax(min_dis) #select the node with the max min distance, namely the node that is most different from the selected nodes
        # print('ind: ', ind)
        assert nodes_idx[ind] not in selected_nodes

        selected_nodes.append(nodes_idx[ind])
        new_batch.append(nodes_idx[ind])
        # print('%d item: %d' %(i, nodes_idx[ind]))
    return np.array(new_batch)


def query_featprop(features, number, nodes_idx):
    features = features.cpu().numpy() #特征传播（Feature Propagation）是一种基于特征相似性的选择策略。通过计算节点特征之间的成对距离，我们可以识别出特征相似的节点，并通过聚类算法选择代表性节点。

    X = features[nodes_idx]
    # print('X: ', X)
    t1 = perf_counter()
    distances = pairwise_distances(X, X)
    print('computer pairwise_distances: {}s'.format(perf_counter() - t1))
    clusters, medoids = k_medoids(distances, k=number) #k-medoids clustering to select the medoids, namely the nodes that are most representative of the clusters
    # print('cluster: ', clusters)
    # print('medoids: ', medoids)
    # print('new indices: ', np.array(nodes_idx)[medoids])
    return np.array(nodes_idx)[medoids]

def query_pr(PR_scores, number, nodes_idx):
    selected_scores = {} #select top n nodes with the highest PageRank scores
    for node in nodes_idx:
        selected_scores[node] = PR_scores[node]
    topk_scores = {k: v for k, v in sorted(selected_scores.items(), key = lambda item: item[1])}
    # print('ppr_scores: ', PPR_scores)
    # print('topk_scores: ', topk_scores)
    # for key in topk_scores.keys():
    #     print('ppr[{}]: {}, PR_scores[{}]: {}'.format(key, PPR_scores[key], key, PR_scores[key]))
    # print('tok_scores: ', [v for k,v in list(topk_scores.items())[-number:]])
    return list(topk_scores.keys())[-number:]

# PageRank 算法的基本思想是：一个节点的重要性取决于指向它的节点的数量和质量。
# 1. 基本公式：
# 对于节点 i，其 PageRank 值 PR(i) 的计算公式为：
# PR(i) = (1-d) + d Σ(PR(j) / C(j))
# 其中：
# d 是阻尼因子（通常设为 0.85）
# j 是指向节点 i 的所有节点
# C(j) 是节点 j 的出度（指向其他节点的边的数量）
# 2. 迭代计算：
# 初始化所有节点的 PageRank 值为 1/N（N 是节点总数）
# 重复应用上述公式，直到 PageRank 值收敛或达到预定的迭代次数