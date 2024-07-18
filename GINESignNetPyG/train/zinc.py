import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import torch
from core.config import cfg, update_cfg
from core.train import run, run_k_fold
from core.model import GNN
from core.sign_net import SignNetGNN, RandomGNN, SingleGNN
from core.transform import EVDTransform
import argparse
from torch_geometric.datasets import ZINC
import optuna
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import degree, is_undirected, contains_self_loops

def check_distinct(data):
    return len(data.eigen_values) == len(torch.unique(data.eigen_values)) 

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = EVDTransform('sym')
    # transform = transform_eval = None
    root = 'data/ZINC/'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    # train_dataset.edge_index is undirected and no self loops
    #laplacian = get_laplacian(train_dataset)
    #train_dataset.edge_attr = torch.cat((train_dataset.edge_attr.unsqueeze(1), laplacian), dim=1)
    val_dataset = ZINC(root, subset=True, split='val', transform=transform_eval) 
    #laplacian = get_laplacian(val_dataset)
    #val_dataset.edge_attr = torch.cat((val_dataset.edge_attr.unsqueeze(1), laplacian), dim=1)
    test_dataset = ZINC(root, subset=True, split='test', transform=transform_eval) 
    #laplacian = get_laplacian(test_dataset)
    #test_dataset.edge_attr = torch.cat((test_dataset.edge_attr.unsqueeze(1), laplacian), dim=1)
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
                           bn=cfg.model.bn,
                           res=cfg.model.res,
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers,
                           exp_after=cfg.model.exp_after)
    elif cfg.model.gnn_type == 'attr':
        model = SingleGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           bn=cfg.model.bn,
                           res=cfg.model.res,
                           gnn_type='SimplifiedPNAConv',
                           nl_gnn=cfg.model.num_layers,
                           exp_after=cfg.model.exp_after)
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
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        if num_samples is not None and num_samples != 'attr':
            if num_samples == 'samples_only':
                add_x = torch.randn(data.x.shape[0], 128, 100).to(torch.int64).to(device)
                loss = (model(data, add_x).squeeze() - y).abs().mean()
            else:
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
            if num_samples == 'samples_only':
                add_x = torch.randn(data.x.shape[0], 128, 100).to(torch.int64).to(device)
                total_error += (model(data, add_x).squeeze() - y).abs().sum().item()
            else:
                rand_x = torch.randn((data.x.shape[0], 1, num_samples)).to(device)
                total_error += (model(data, rand_x).squeeze() - y).abs().sum().item()
        else:
            total_error += (model(data).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf

def save_visualizations(study, cfg, log_dir):
    # Plot optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"{log_dir}/optimization_history.png")

    # Custom visualization of training and testing values for each epoch
    best_trial = study.best_trial
    epochs = range(1, cfg.train.epochs + 1)

    test_values = [best_trial.user_attrs[f'epoch_{epoch}_test'] for epoch in epochs]
    train_values = [best_trial.user_attrs[f'epoch_{epoch}_train'] for epoch in epochs]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, test_values, label='Test Performance')
    plt.plot(epochs, train_values, label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Performance')
    plt.legend()
    plt.title('Train and Test Performance over Epochs')
    plt.savefig(f"{log_dir}/train_test_performance.png")
    plt.close()

    # Custom scatter plot of objective values and hyperparameters
    trial_numbers = range(1, len(study.trials) + 1)
    learning_rates = [t.params['learning_rate'] for t in study.trials]
    weight_decays = [t.params['weight_decay'] for t in study.trials]
    num_layers = [t.params['num_layers'] for t in study.trials]
    objective_values = [t.value for t in study.trials]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].scatter(trial_numbers, objective_values, c=learning_rates, cmap='viridis')
    ax[0].set_xlabel('Trial')
    ax[0].set_ylabel('Objective Value')
    ax[0].set_title('Learning Rate')
    cbar = fig.colorbar(ax[0].collections[0], ax=ax[0])
    cbar.set_label('Learning Rate')

    ax[1].scatter(trial_numbers, objective_values, c=weight_decays, cmap='viridis')
    ax[1].set_xlabel('Trial')
    ax[1].set_ylabel('Objective Value')
    ax[1].set_title('Weight Decay')
    cbar = fig.colorbar(ax[1].collections[0], ax=ax[1])
    cbar.set_label('Weight Decay')

    ax[2].scatter(trial_numbers, objective_values, c=num_layers, cmap='viridis')
    ax[2].set_xlabel('Trial')
    ax[2].set_ylabel('Objective Value')
    ax[2].set_title('Number of Layers')
    cbar = fig.colorbar(ax[2].collections[0], ax=ax[2])
    cbar.set_label('Number of Layers')

    plt.savefig(f"{log_dir}/hyperparameters_scatter.png")
    plt.close()
    try:
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f"{log_dir}/param_importances.png")
    except Exception as e:
        print(f"Error plotting param importances")

def log_results(study, log_file):
    # Sort trials by their objective value in descending order (maximize)
    sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
    
    with open(log_file, 'w') as f:
        f.write(f'Best hyperparameters: {study.best_params}\n')
        f.write(f'Best validation performance: {study.best_value}\n')

        f.write('\nTop 5 Trials:\n')
        for rank, trial in enumerate(sorted_trials[:5], start=1):
            f.write(f'\nTrial {rank}:\n')
            f.write(f'Number: {trial.number}\n')
            f.write(f'Value: {trial.value}\n')
            f.write(f'Params: {trial.params}\n')

            epochs = range(1, 310)  # Use the configured number of epochs
            
            f.write('\nTrain and Test Performance per Epoch:\n')
            for epoch in epochs:
                try:
                    train_val = trial.user_attrs[f'epoch_{epoch}_train']
                    test_val = trial.user_attrs[f'epoch_{epoch}_test']
                    f.write(f'Epoch {epoch}: Train Loss = {train_val}, Test Performance = {test_val}\n')
                except KeyError as e:
                    f.write(f"Missing value for epoch {epoch}: {e}\n")
                    print(f"Missing value for epoch {epoch}: {e}")


def objective(trial, cfg):
    # Define the hyperparameters to search
    cfg.train.lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    cfg.model.num_layers = trial.suggest_int('num_layers', 5, 7, step=1)
    cfg.model.pool = trial.suggest_categorical('pool', ['add', 'mean'])

    cfg.model.hidden_size = trial.suggest_int('hidden_size', 70, 130, 180)
    cfg.model.res = trial.suggest_categorical('res', [True, False])
    cfg.model.nlayer_inner = trial.suggest_int('num_layers',1, 2, 3)
    cfg.model.bn = trial.suggest_categorical('BN', ['BN', False])
    cfg.model.exp_after = trial.suggest_categorical('exp_after', [True, False])
    
    try:
        mean_perf, avg_test_curve, avg_train_curve = run_k_fold(cfg, create_dataset, create_model, train, test, trial, evaluator=None, k=2)
    except optuna.exceptions.TrialPruned:
        raise optuna.TrialPruned()
    for epoch, (test_val, train_val) in enumerate(zip(avg_test_curve, avg_train_curve)):
        trial.set_user_attr(f"epoch_{epoch+1}_test", test_val)
        trial.set_user_attr(f"epoch_{epoch+1}_train", train_val)
    
    return mean_perf


def create_mock_study():
    def objective(trial):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
        num_layers = trial.suggest_categorical('num_layers', [5, 6, 7])
        for epoch in range(1, 21):
            trial.set_user_attr(f"epoch_{epoch}_test", 1.0 / epoch)
            trial.set_user_attr(f"epoch_{epoch}_train", 1.0 / (epoch + 0.5))
        return 1.0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    return study

# Create a mock cfg object with necessary attributes
class MockCfg:
    class Train:
        epochs = 20
    train = Train()


if __name__ == '__main__':
    # get config 
    # Create mock study and cfg
        #mock_study = create_mock_study()
        #mock_cfg = MockCfg()
        #save_visualizations(mock_study, mock_cfg, '/lfs/hyperturing1/0/echoi1/Random-SignNet/GINESignNetPyG')
        #log_results(mock_study, f"/lfs/hyperturing1/0/echoi1/Random-SignNet/GINESignNetPyG/results.txt")
    parser = argparse.ArgumentParser(description="Run training with specified config file")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--k_fold', type=bool, default=False, help="Whether to run kfold hyperparam search")
    args = parser.parse_args()
    config_path = args.config

    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_path)
    cfg = update_cfg(cfg)
    if args.k_fold:
        pruner = optuna.pruners.MedianPruner(5,70,2)
        study = optuna.create_study(direction='maximize', pruner=pruner)
        study.optimize(lambda trial: objective(trial, cfg), n_trials=40)
        save_visualizations(study, cfg, '/lfs/hyperturing1/0/echoi1/Random-SignNet/GINESignNetPyG')
        log_results(study, f"/lfs/hyperturing1/0/echoi1/Random-SignNet/GINESignNetPyG/results_val_2.txt")
    else:
        run(config_path, cfg, create_dataset, create_model, train, test)
