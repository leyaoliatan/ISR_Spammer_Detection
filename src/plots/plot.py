import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle as pkl
import os
import datetime
import json
import pandas as pd
from torch_geometric.utils import get_laplacian
from torch_geometric.data import HeteroData
from torch_geometric.utils import dropout_adj
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

plt.rcParams.update({
    "figure.figsize": (6, 4),
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
})
plt.style.use('seaborn-whitegrid')

def plot(index1, data1, index2, data2, index3, data3, figure_name):
    # Create the plot
    plt.figure()
    plt.plot(index1, data1, label='Our Method', color='steelblue', linestyle='-')
    plt.plot(index2, data2, label='Wu et al., 2023', color='#2ca02c', linestyle='-')
    plt.plot(index3, data3, label='Our Method \n without Further Training', color='steelblue', linestyle='--')
    #x-axis
    plt.xticks(np.arange(0, 4500, 500))
    
    # Annotations for data1
    #plt.annotate(f'Min: {min(data1):.2f}', xy=(index1[np.argmin(data1)], min(data1)-0.01), 
                #  xytext=(5, 5), textcoords='offset points', fontsize=10, color='#1f77b4')
    plt.annotate(f'Best: {max(data1):.2f}', xy=(index1[np.argmax(data1)]-200, max(data1)-0.01), 
                 xytext=(5, -10), textcoords='offset points', fontsize=10, color='#1f77b4')
    plt.annotate(f'Last: {data1[-1]:.2f}', xy=(index1[-1]-1, data1[-1]-0.02), 
                 xytext=(-20, -20), textcoords='offset points', fontsize=10, color='#1f77b4')

    # Annotations for data2
    #plt.annotate(f'Min: {min(data2):.2f}', xy=(index2[np.argmin(data2)], min(data2)-0.01), 
                #  xytext=(5, 5), textcoords='offset points', fontsize=10, color='#2ca02c')
    plt.annotate(f'Best: {max(data2):.2f}', xy=(index2[np.argmax(data2)]-500, max(data2)-0.01), 
                 xytext=(5, -10), textcoords='offset points', fontsize=10, color='#2ca02c')
    plt.annotate(f'Last: {data2[-1]:.2f}', xy=(index2[-1]-100, data2[-1]-0.02), 
                 xytext=(-20, -20), textcoords='offset points', fontsize=10, color='#2ca02c')

    # Axis labels, title, and legend
    plt.xlabel('Number of Nodes in Training Set Used')
    plt.ylabel('F1-Score')
    plt.title("AmazonCN: Comparison of Method Performance")
    plt.legend()

    # Save the plot
    plt.tight_layout()
    plt.savefig(f"./figures/{figure_name}.png", dpi=300, bbox_inches='tight')
    plt.close()


f1_all, f1_test = [], []
data_path = "/Users/leahtan/Documents/3_Research/2024-Ali/ISR/results/loss_5percent_global.txt"
with open(data_path, 'r') as file:
    for line in file:
        if 'f1_test_all:' in line and 'f1_test:' in line:
            data = line.split(", ")
            f1_all.append(float(data[1].split(": ")[1]))
            f1_test.append(float(data[3].split(": ")[1]))

# Plotting data with enhanced readability and alignment to CS publication style
num_train = [i*9424 for i in [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.3, 0.5]]
plot(
    index1=[i * 10 for i in range(40)], data1=f1_all, 
    index2=num_train, data2=[0.707, 0.737, 0.726, 0.788, 0.821, 0.845, 0.852, 0.863], 
    index3=[(i + 40) * 10 for i in range(int(4712 / 10) - 40)], 
    data3=[0.929 for _ in range(int(4712 / 10) - 40)], 
    figure_name="separate_test_compare"
)
# plot(
#     index1=[i * 10 for i in range(40)], data1=f1_test, 
#     index2=num_train, data2=[0.815, 0.891, 0.898, 0.899], 
#     index3=[(i + 40) * 10 for i in range(int((7959 + 400 - 4187) / 10) - 40)], 
#     data3=[0.926 for _ in range(int((7959 + 400 - 4187) / 10) - 40)], 
#     figure_name="intersection_test_compare"
# )
