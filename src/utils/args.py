#src/utils/args.py
import argparse
import torch

def get_citation_args():
    parser = argparse.ArgumentParser(description='Experiments on using active learning to detect community spammer')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')

    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train.')
    
    parser.add_argument('--lr', type=float, default=0.2,
                        help='Initial learning rate.')
    
    parser.add_argument('--weight_decay', type=float, default=5e-6,
                        help='Weight decay (L2 loss on parameters).') # should use 5e-6 for our method
    
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units.') #Q, in gin_adv, hidden_dim is set to 32 as defualt, and in the original code, args.hidden seems not transfered into the model, so the actual hidden_dim is 32?
    
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate (1 - keep probability).') #Q, in gin_adv, there's no dropout argument
    
    parser.add_argument('--dataset', type=str, default="amazon",
                        help='Dataset to use.')
    
    parser.add_argument('--model', type=str, default="GIN_adv",
                        choices=["GCN","GCN_adv","GIN","GIN_adv","GAT"],
                        help='Graph model to use. Models are stored in models.py')
    
    parser.add_argument('--feature', type=str, default="non",
                        choices=['non', 'mul', 'cat', 'adj'],
                        help='feature-type')
    
    parser.add_argument('--normalization', type=str, default='AugNormAdj',
                       choices=['AugNormAdj'],
                       help='Normalization method for the adjacency matrix.')
    
    parser.add_argument('--degree', type=int, default=2,
                        help='degree of the approximation.')
    
    # parser.add_argument('--per', type=int, default=-1,
    #                     help='Number of each nodes so as to balance.')
    
    # parser.add_argument('--experiment', type=str, default="base-experiment",
    #                     help='feature-type')
    
    # parser.add_argument('--tuned', action='store_true', help='use tuned hyperparams')

    parser.add_argument('--strategy', type=str, default='uncertainty', help='query strategy, default is uncertainty')
    parser.add_argument('--train_size', type=float, default=0.1, help='Proportion of train set to all labeled set (default 10%)')

    parser.add_argument('--file_io', type=int, default=True,
                        help='determine whether use file io to use seeds, the seeds txt name is random_seed.txt')
    
    parser.add_argument('--reweight', type=int, default=1,
                        choices=[0, 1],
                        help='whether to use reweighting')
    
    parser.add_argument('--adaptive', type=int, default=1,
                        choices=[0, 1],
                        help='to use adaptive weighting')
    
    parser.add_argument('--lambdaa', type=float, default=0.99,
                        help='control combination')
    
    parser.add_argument('--version_name', type=str, default="v1",
                        help='the name of the saved figure, default is v1')
    
    parser.add_argument('--save_pred',type = bool, default=True,
                        help = 'store the prediction results on all unsed nodes')

    parser.add_argument('--data_path', type=str, default='/Users/leahtan/Documents/3_Research/2024-Ali/ISR/data/raw', help='data path')
    parser.add_argument('--save_path', type=str, default='/Users/leahtan/Documents/3_Research/2024-Ali/ISR/results', help='save path')



    args, _ = parser.parse_known_args()
    args.cuda = args.cuda and torch.cuda.is_available() #if so, cuda is available
    return args
