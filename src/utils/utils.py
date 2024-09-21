import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
import datetime
import os
import pandas as pd
import networkx as nx
from community import community_louvain
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import os

def plot(index, data_list, figure_name, labels=None, title=None, xlabel='Index', ylabel='Value', save_path="./figures/"):
    """
    绘制多个预测结果的折线图，并进行标注，适用于学术格式。

    参数：
    - index (list or array): x 轴的索引值。
    - data_list (list of lists or arrays): 多个预测结果的数据，每个内部列表或数组对应一条折线。
    - figure_name (str): 图表的名称，用于保存文件时命名。
    - labels (list of str, 可选): 每个预测结果对应的标签，用于图例显示。长度应与 data_list 相同。
    - title (str, 可选): 图表的标题。如果未提供，将使用 figure_name 作为标题。
    - xlabel (str, 可选): x 轴标签，默认为 'Index'。
    - ylabel (str, 可选): y 轴标签，默认为 'Value'。
    - save_path (str, 可选): 图表保存的目录路径，默认为 "./figures/"。

    返回：
    - None
    """

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制每一条数据线
    for idx, data in enumerate(data_list):
        label = labels[idx] if labels and idx < len(labels) else f'Data {idx+1}'
        plt.plot(index, data, label=label, linewidth=2)

        # 找出最小值、最大值和最后的值
        min_value = min(data)
        max_value = max(data)
        last_value = data[-1]
        min_index = data.index(min_value)
        max_index = data.index(max_value)
        last_index = len(data) - 1

        # 标注最小值
        plt.annotate(f'Min: {min_value:.2f}',
                     xy=(index[min_index], min_value),
                     xytext=(index[min_index], min_value),
                     textcoords='data',
                     fontsize=10,
                     color=plt.gca().lines[idx].get_color())

        # 标注最大值
        plt.annotate(f'Max: {max_value:.2f}',
                     xy=(index[max_index], max_value),
                     xytext=(index[max_index], max_value),
                     textcoords='data',
                     fontsize=10,
                     color=plt.gca().lines[idx].get_color())

        # 标注最后的值
        # plt.annotate(f'Last: {last_value:.2f}',
        #              xy=(index[last_index], last_value),
        #              xytext=(index[last_index], last_value),
        #              textcoords='data',
        #              fontsize=10,
        #              color=plt.gca().lines[idx].get_color())

    # 设置标题和轴标签
    plt.title(title if title else figure_name, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)

    # 显示图例
    if labels:
        plt.legend(fontsize=12)
    else:
        plt.legend(fontsize=12)
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.5)

    # 优化布局
    plt.tight_layout()

    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存图表
    plt.savefig(os.path.join(save_path, f"{figure_name}.png"), dpi=300)
    plt.close()



def print_graph_info(G):
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is directed: {G.is_directed()}")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")


def augment_feature(feature, nx_G):
    print("===== 1. The modularity-based feature augmentation. =====")
    partition = community_louvain.best_partition(nx_G)
    modularity = community_louvain.modularity(partition, nx_G)
    print(f"the modularity of community is {modularity}") #this gives the modularity of the whole graph
    # 创建一个字典存储每个社区的modularity值
    node_modularity = {}
    for community in set(partition.values()):
        # 取出该社区的节点
        nodes_in_community = [node for node, comm in partition.items() if comm == community]
        # 计算该社区在整体中的modularity贡献
        subgraph = nx_G.subgraph(nodes_in_community)
        # print(subgraph)
        community_partition = {node: community for node in nodes_in_community}
        community_modularity = community_louvain.modularity({**partition, **community_partition}, nx_G)
        # 分配给该社区中的每个节点
        for node in nodes_in_community:
            node_modularity[node] = community_modularity
    
    augmented_mod_feat = []
    for i in range(feature.shape[0]):
        if i in node_modularity:
            augmented_mod_feat.append(node_modularity[i])
        else:
            augmented_mod_feat.append(0)
    # kcore based 

    augmented_core_feat = []
    print("===== 2. The k-core-based feature augmentation. =====")
    # Calculate k-core values for each node
    core_numbers = nx.core_number(nx_G)
    #print the max core number
    print(f"the max core number is {max(core_numbers.values())}")
    for i in range(feature.shape[0]):
        if i in core_numbers:
            augmented_core_feat.append(core_numbers[i])
        else:
            augmented_core_feat.append(0)
    
    # print(augmented_core_feat)
    result = np.column_stack((feature, np.array(augmented_mod_feat), np.array(augmented_core_feat)))
    return result

def sgc_precompute(features, adj, degree):
    t = perf_counter()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = perf_counter()-t
    return features, precompute_time

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda: torch.cuda.manual_seed(seed)

def get_classes_statistic(ally):
    """将每个类别的计数转换为比例（占总样本的百分比）"""
    classes_dict = defaultdict(int)
    # ally = np.argmax(ally, axis=1)  # to index
    for y in ally:
        classes_dict[y] += 1
    classes_dict = dict(classes_dict)
    for k in classes_dict.keys():
        classes_dict[k] = classes_dict[k] / len(ally)
    # return sorted(classes_dict.items(), key= lambda x:(x[1]))
    return classes_dict

#test plot
if __name__ == '__main__':
    # 生成模拟数据
    index = list(range(1, 21))  # x 轴从1到20

    # 模拟三组预测结果
    data1 = [0.5 + 0.02 * x**2 for x in index]           
    data2 = [0.6 + 0.015 * x**2 for x in index]           
    data3 = [0.4 + 0.025 * x**2 for x in index]           

    data_list = [data1, data2, data3]
    labels = ['A', 'B', 'C']
    figure_name = 'test_plot_multiple_models'
    title = 'test_plot_multiple_models'
    xlabel = 'testx'
    ylabel = 'testy'

    # 调用绘图函数
    plot(index, data_list, figure_name, labels=labels, title=title, xlabel=xlabel, ylabel=ylabel)

    print(f"saved to'./figures/{figure_name}.png'")