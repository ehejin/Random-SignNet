import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import torch
from core.config import cfg, update_cfg
from core.train import run, run_k_fold
from core.model import GNN
from core.sign_net import SignNetGNN, RandomGNN, SingleGNN, RandomGNNGraph, GAE
from core.transform import EVDTransform, LAPTransform
import argparse
from torch_geometric.datasets import ZINC, TUDataset
import optuna
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import degree, is_undirected, contains_self_loops
from sklearn.metrics import roc_auc_score
import torch_geometric.utils as pyg_utils


def check_distinct(data):
    return len(data.eigen_values) == len(torch.unique(data.eigen_values)) 


def create_dataset_TUD(cfg): 
    # e.g. cfg.dataset = data/REDDIT-MULTI-5K/
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = EVDTransform('sym')
    root = cfg.dataset
    dataset = TUDataset(root, name=root[5:-1], transform=transform)
    dataset = dataset.shuffle()
    length = len(dataset)
    train_dataset = dataset[:int(0.7*length)]
    val_dataset = dataset[int(0.7*length):int(0.7*length)+int(0.1*length)]
    test_dataset = dataset[int(0.7*length)+int(0.1*length):]
    return train_dataset, val_dataset, test_dataset

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    if cfg.model.transform == 'Laplacian':
        transform = transform_eval = LAPTransform('sym')
    elif cfg.model.transform is None:
        transform = transform_eval = None
    else:
        transform = transform_eval = EVDTransform('sym')
    if cfg.dataset == 'ZINC':
        root = 'data/ZINC/'
        train_dataset = ZINC(root, subset=True, split='train', transform=transform)
        # train_dataset.edge_index is undirected and no self loops
        val_dataset = ZINC(root, subset=True, split='val', transform=transform_eval) 
        test_dataset = ZINC(root, subset=True, split='test', transform=transform_eval) 
    else:
        root = cfg.dataset
        dataset = TUDataset(root, name=root[5:-1], transform=transform)
        dataset = dataset.shuffle()
        length = len(dataset)
        train_dataset = dataset[:int(0.7*length)]
        val_dataset = dataset[int(0.7*length):int(0.7*length)+int(0.1*length)]
        test_dataset = dataset[int(0.7*length)+int(0.1*length):]
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
    model_emb = None
    if cfg.embed_model is not None:
        if cfg.embed_model.gnn_type == 'gae':
            model_emb = GAE(None, None,
                            n_hid=cfg.embed_model.hidden_size, 
                            n_out=cfg.embed_model.nout, 
                            bn=cfg.embed_model.bn,
                            res=cfg.embed_model.res,
                            nl_signnet=1, 
                            nl_gnn=cfg.embed_model.num_layers,
                            exp_after=cfg.embed_model.exp_after, 
                            pooling=cfg.embed_model.pool,
                            laplacian=cfg.embed_model.laplacian,
                            EMBED_SIZE=cfg.embed_model.EMBED_SIZE)
            model_emb.load_state_dict(torch.load(cfg.embed_model.path))
            model_emb.to(torch.device('cpu'))
            model_emb.eval()
        else:
            model_emb = RandomGNN(None, None,
                            n_hid=cfg.embed_model.hidden_size, 
                            n_out=cfg.embed_model.nout, 
                            bn=cfg.embed_model.bn,
                            res=cfg.embed_model.res,
                            nl_signnet=1, 
                            nl_gnn=cfg.embed_model.num_layers,
                            exp_after=cfg.embed_model.exp_after, 
                            pooling=cfg.embed_model.pool,
                            laplacian=cfg.embed_model.laplacian,
                            EMBED_SIZE=cfg.embed_model.EMBED_SIZE,
                            regression_type=cfg.regression) # match used during training
            model_emb.load_state_dict(torch.load(cfg.embed_model.path))
            model_emb.to(torch.device('cpu'))
            model_emb.eval()
    if cfg.model.gnn_type == 'SignNet':
        model = SignNetGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers)
    elif cfg.model.gnn_type == 'Random':
        print("RANDOM")
        model = RandomGNNGraph(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           bn=cfg.model.bn,
                           res=cfg.model.res,
                           embedding_size=cfg.model.embedding_size,
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers,
                           exp_after=cfg.model.exp_after, 
                           pooling=cfg.model.pool,
                           laplacian=cfg.model.laplacian)
    elif cfg.model.gnn_type == 'samples_only':
        print("ONLY RANDOM")
        model = RandomGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=cfg.model.nout, 
                           bn=cfg.model.bn,
                           res=cfg.model.res,
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers,
                           exp_after=cfg.model.exp_after, 
                           pooling=cfg.model.pool,
                           laplacian=cfg.model.laplacian,
                           EMBED_SIZE=cfg.model.EMBED_SIZE,
                           regression_type=cfg.regression)
    elif cfg.model.gnn_type == 'attr':
        model = SingleGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           bn=cfg.model.bn,
                           res=cfg.model.res,
                           gnn_type='SimplifiedPNAConv',
                           nl_gnn=cfg.model.num_layers,
                           exp_after=cfg.model.exp_after)
    elif cfg.model.gnn_type == 'gae':
        model = GAE(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=cfg.model.nout, 
                           bn=cfg.model.bn,
                           res=cfg.model.res,
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers,
                           exp_after=cfg.model.exp_after, 
                           pooling=cfg.model.pool,
                           laplacian=cfg.model.laplacian,
                           EMBED_SIZE=cfg.model.EMBED_SIZE)
    else:
        model = GNN(None, None, 
                    nhid=cfg.model.hidden_size, 
                    nout=1, 
                    nlayer=cfg.model.num_layers, 
                    gnn_type=cfg.model.gnn_type, 
                    dropout=cfg.train.dropout, 
                    pooling=cfg.model.pool,
                    res=True)
    return model, model_emb

def normalize_tensor_to_range_zero_to_two(X):
    X = X.float()
    X_min = torch.min(X)
    X_max = torch.max(X)
    X_normalized = (X - X_min) / (X_max - X_min)
    X_scaled = 2 * X_normalized
    
    return X_scaled

def get_node_embedding(data, normalize, model_emb, num_samples):
    add_x = torch.randn(data.x.shape[0], num_samples, 1).to(torch.int64).to(torch.device('cuda'))
    with torch.no_grad():
        try:
            embed = model_emb(data, add_x, get_embeddings=True)
            if embed.dim() > 2:
                embed = embed.mean(-2)
        except:
            embed = model_emb(data, add_x)
            if embed.dim() > 2:
                embed = embed.mean(-2)
    if normalize:
        embed = normalize_tensor_to_range_zero_to_two(embed)
    return embed

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def train(train_loader, model, model_emb, optimizer, device, num_samples, get_node_emb=False, 
          hidden_size=0, regression=True, train_gae=True, normalize=False):
    total_loss = 0
    N = 0 
    if regression == 'B' or train_gae:
        criterion = torch.nn.BCEWithLogitsLoss()
    elif regression == 'M':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = None
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        if get_node_emb:
            model.to(torch.device('cpu')) 
            model_emb.to(torch.device('cuda'))
            embed = get_node_embedding(data, normalize, model_emb, num_samples)
            data.x = torch.cat((data.x, embed.to(torch.int64)), dim=-1)
            model_emb.to(torch.device('cpu'))
            model.to(torch.device('cuda'))
        optimizer.zero_grad()
        if num_samples is not None:
            add_x = torch.randn(data.num_nodes, num_samples, 1).to(torch.int64).to(device)
            if regression == 'R' and not train_gae: 
                loss = (model(data, add_x).squeeze() - y).abs().mean()
            elif train_gae:
                out = model(data, add_x)
                adj_matrix = pyg_utils.to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0]
                loss = criterion(out.float(), adj_matrix.to(device).float()) 
            else:
                out = model(data, add_x)
                loss = criterion(out.float(), y.long())                    
        else:
            loss = (model(data).squeeze() - y).abs().mean()
        with torch.autograd.set_detect_anomaly(True):
            loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test(loader, model, model_emb, evaluator, device, num_samples, get_node_emb=False, 
         hidden_size=0, regression='R', train_gae=False, normalize=False):
    if train_gae:
            # Variables to store metrics
            y_true, y_pred = [], []

            for data in loader:
                data = data.to(device)
                add_x = torch.randn(data.num_nodes, num_samples, 1).to(torch.int64).to(device)
                output = model(data, add_x)
                pred_adj = output
                true_adj = pyg_utils.to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes)[0].flatten()
                pred_adj = pred_adj.flatten()
                y_true.append(true_adj.cpu().numpy())
                y_pred.append(pred_adj.cpu().numpy())

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            test_perf = roc_auc_score(y_true, y_pred)
    else:
        total_error = 0
        N = 0
        total = 0
        correct = 0
        for data in loader:
            if isinstance(data, list):
                data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
            else:
                data, y, num_graphs = data.to(device), data.y, data.num_graphs
            if get_node_emb:
                model.to(torch.device('cpu'))
                model_emb.to(torch.device('cuda'))
                embed = get_node_embedding(data, normalize, model_emb, num_samples)
                data.x = torch.cat((data.x, embed.to(torch.int64)), dim=-1)
                model_emb.to(torch.device('cpu'))
                model.to(torch.device('cuda'))
            if num_samples is not None:
                add_x = torch.randn(data.num_nodes, num_samples, 1).to(torch.int64).to(device)
                #add_x = torch.randn(data.x.shape[0], hidden_size, 80).to(torch.int64).to(device)
                if regression == 'R':
                    total_error += (model(data, add_x).squeeze() - y).abs().sum().item()
                elif regression == 'M':
                    outputs = model(data, add_x)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == data.y).sum().item() 
                    total += y.size(0)
                else:
                    outputs = model(data, add_x)
                    predicted = (outputs > 0.5).squeeze()  
                    correct += (predicted == data.y).sum().item()
                    total += data.y.size(0)
            else:
                total_error += (model(data).squeeze() - y).abs().sum().item()
            N += num_graphs
        if regression == "R":
            test_perf = - total_error / N
        else: 
            test_perf = correct / total
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
    #cfg.train.lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    cfg.model.num_layers = trial.suggest_int('num_layers', 2, 4, step=1)#8, 10, step=1)
    #cfg.model.pool = trial.suggest_categorical('pool', ['add', 'mean'])
    if cfg.model.num_layers == 10:
        cfg.model.hidden_size = trial.suggest_int('hidden_size', 20, 70, step=25) # 240, 360, step=30) #
    #elif cfg.model.num_layers == 3:
    #    cfg.model.hidden_size = trial.suggest_int('hidden_size', 210, 250, step=35)
    else:
        cfg.model.hidden_size = trial.suggest_int('hidden_size', 20, 80, step=20) # 160, 186, step=25)#
    #cfg.model.res = trial.suggest_categorical('res', [True, False])
    #cfg.model.nlayer_inner = trial.suggest_int('num_layers',1, 2, 3)
    #cfg.model.bn = trial.suggest_categorical('BN', ['BN', False])
    #cfg.model.exp_after = trial.suggest_categorical('exp_after', [True, False])
    print(cfg.model.num_layers, cfg.model.hidden_size)
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
        study.optimize(lambda trial: objective(trial, cfg), n_trials=7)
        #save_visualizations(study, cfg, '/lfs/hyperturing1/0/echoi1/Random-SignNet/GINESignNetPyG')
        log_results(study, f"./results_MORE_layers2.txt")
    else:
        run(config_path, cfg, create_dataset, create_model, train, test, model_path=cfg.model_path, 
            log_wandb=cfg.wandb)
