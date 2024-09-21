import yaml
import os
import torch

def load_config(config_path="/Users/leahtan/Documents/3_Research/2024-Ali/ISR/configs/default_config.yaml"):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_device(config):
    if config['environment']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['environment']['cuda_device']}")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device