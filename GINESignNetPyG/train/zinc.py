import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import torch
from core.config import cfg, update_cfg
from core.train import run, run_k_fold
from core.model import GNN
from core.sign_net import SignNetGNN, RandomGNN
from core.transform import EVDTransform

from torch_geometric.datasets import ZINC
import argparse
import optuna
import csv
from functools import partial

def check_distinct(data):
    return len(data.eigen_values) == len(torch.unique(data.eigen_values)) 

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = EVDTransform('sym')
    # transform = transform_eval = None
    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    val_dataset = ZINC(root, subset=True, split='val', transform=transform_eval) 
    test_dataset = ZINC(root, subset=True, split='test', transform=transform_eval)   

    train_num_distincts = [check_distinct(data) for data in train_dataset]
    val_num_distincts = [check_distinct(data) for data in val_dataset]
    test_num_distincts = [check_distinct(data) for data in test_dataset]
    print(f"Percentage of graphs with distinct eigenvalues (train): {100*sum(train_num_distincts)/len(train_num_distincts)}%")
    print(f"Percentage of graphs with distinct eigenvalues (vali): {100*sum(val_num_distincts)/len(val_num_distincts)}%")
    print(f"Percentage of graphs with distinct eigenvalues (test): {100*sum(test_num_distincts)/len(test_num_distincts)}%")

    return train_dataset, val_dataset, test_dataset

def create_dataset_kfold(cfg): 
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = EVDTransform('sym')
    # transform = transform_eval = None
    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    train_num_distincts = [check_distinct(data) for data in train_dataset]
    print(f"Percentage of graphs with distinct eigenvalues (train): {100*sum(train_num_distincts)/len(train_num_distincts)}%")
    return train_dataset

def create_model(cfg):
    if cfg.model.gnn_type == 'SignNet':
        model = SignNetGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers)
    elif cfg.model.gnn_type == 'Random':
        print("RANDOM")
        model = RandomGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers)
    else:
        model = GNN(None, None, 
                    nhid=cfg.model.hidden_size, 
                    nout=1, 
                    nlayer=cfg.model.num_layers, 
                    gnn_type=cfg.model.gnn_type, 
                    dropout=cfg.train.dropout, 
                    pooling=cfg.model.pool,
                    res=True)

    return model


def train(train_loader, model, optimizer, device, num_samples):
    total_loss = 0
    N = 0 
    #with torch.profiler.profile(activities=[
    #    torch.profiler.ProfilerActivity.CPU,
    #    torch.profiler.ProfilerActivity.CUDA,
    #],
    #profile_memory=True,  # This will track memory usage
    #record_shapes=True,   # This will track tensor shapes
    #with_stack=True       # This will include stack traces
    #) as prof:
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        if num_samples is not None:
            rand_x = torch.randn((data.x.shape[0], 1, num_samples)).to(device)
            loss = (model(data, rand_x).squeeze() - y).abs().mean()
        else:
            loss = (model(data).squeeze() - y).abs().mean()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    #        break
    #print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
    #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device, num_samples):
    total_error = 0
    N = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        if num_samples is not None:
            rand_x = torch.randn((data.x.shape[0], 1, num_samples)).to(device)
            total_error += (model(data, rand_x).squeeze() - y).abs().sum().item()
        else:
            total_error += (model(data).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf

hyperparameter_combinations = [
    {'bn': None},
    {'bn': 'batchnorm'},
    {'bn': 'layernorm'}
]
'''hyperparameter_combinations = [
    {'exp_after_regression': True},
    {'exp_after_regression': False},
]'''
'''hyperparameter_combinations = [
    {'pna_layers': 4},
    {'pna_layers': 6},
]'''

def objective(trial, cfg, hyperparams):
    bn = hyperparams['bn']
    #exp_a_r = hyperparams['exp_after_regression']
    #pna_layers = hyperparams['pna_layers']

    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)
    
    cfg.bn = bn
    #cfg.exp_after_regression = exp_a_r
    #cfg.pna_layers = pna_layers
    #run_name = 'pna:' + str(pna_layers)
    run_name = 'norm:' + str(bn)
    #run_name = 'EAR:' + str(exp_a_r)

    test_perf_mean = run_k_fold(run_name, cfg, create_dataset, create_model, train, test, k=2)
    #print(bn, test_perf_mean)
    with open('hyperparam_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([run_name, test_perf_mean])
    
    return test_perf_mean


if __name__ == '__main__':
    # get config 
    parser = argparse.ArgumentParser(description="Run training with specified config file")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    config_path = args.config

    cfg.set_new_allowed(True) 
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)
    run(config_path, cfg, create_dataset, create_model, train, test)
    #run_k_fold(cfg, create_dataset_kfold, create_model, train, test, evaluator=None, k=10)
    '''for hyperparams in hyperparameter_combinations:
        trial = optuna.trial.FixedTrial(hyperparams)
        objective(trial, cfg, hyperparams)'''
