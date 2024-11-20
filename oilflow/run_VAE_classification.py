"Variatioanl AutoEncoder Model"

import os
import random
import numpy as np
import matplotlib.pylab as plt
import urllib.request
import tarfile
import timeit

import torch
from torch.utils.data import TensorDataset, DataLoader
import tqdm

# gpytorch imports
import sys
sys.path.insert(0,'../')
from util.qVAE import *

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

q = 2.0
POWER = torch.tensor(q, device=device)

# train module
def train_VAE(train_loader, model, num_epochs=10000, save_model=False, **kwargs):
    
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002) # 2:0.002, 1.0:0.005, 1.5:0.003, 1.15:0.004
    
    loss_list = []
    f_name = 'oilflow_vae_q'+str(q)+'_class_'+kwargs.pop('class_','full')+'_checkpoint_seed'+str(kwargs.pop('seed',2024))+'.dat'
    if os.path.exists(os.path.join('./results/cls',f_name)):
        state_dict = torch.load(os.path.join('./results/cls',f_name), map_location=device)
        # load the best model
        model.load_state_dict(state_dict['model'])
        min_loss = state_dict['loss']
    else:
        # set device
        model = model.to(device)
        
        # Training loop - optimises the objective wrt variational params using the optimizer provided.
        model.train()
        
        os.makedirs('./results/cls', exist_ok=True)
        # num_epochs = 10000
        iterator = tqdm.tqdm(range(num_epochs), desc="Epoch")
        for epoch in iterator:
            minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
            loss_i = 0
            for x, _ in minibatch_iter:
                if torch.cuda.is_available():
                    x = x.cuda()
                optimizer.zero_grad()
                x_hat, mean, log_var = model(x)
                loss = model.loss(x, x_hat, mean, log_var)
                loss.backward()
                optimizer.step()
                loss_i += loss.item()
                # minibatch_iter.set_postfix(loss=loss.item())
            loss_i /= len(minibatch_iter)
            # record the loss and the best model
            loss_list.append(loss_i)
            if epoch==0:
                min_loss = loss_list[-1]
                optim_model = model.state_dict()
            else:
                if loss_list[-1] < min_loss:
                    min_loss = loss_list[-1]
                    optim_model = model.state_dict()
            print('Epoch {}/{}: Loss: {}'.format(epoch, num_epochs, loss_i ))
        if save_model:
            # save the model
            model_state_dict = optim_model#.state_dict()
            torch.save({'model': model_state_dict, 'loss': min_loss}, os.path.join('./results/cls',f_name))
    # ready for evaluation
    model.eval()
    
    return model, min_loss

def main(seed=2024):
    # Setting manual seed for reproducibility
    # seed=2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load data
    if not (os.path.exists('DataTrn.txt') and os.path.exists('DataTrnLbls.txt')):
        # download data
        url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
        urllib.request.urlretrieve(url, '3PhData.tar.gz')
        with tarfile.open('3PhData.tar.gz', 'r') as f:
            f.extract('DataTrn.txt')
            f.extract('DataTrnLbls.txt')
    
    Y = torch.Tensor(np.loadtxt(fname='DataTrn.txt'))
    labels = torch.Tensor(np.loadtxt(fname='DataTrnLbls.txt'))
    labels = (labels @ np.diag([0, 1, 2])).sum(axis=1).int()
    batch_size = 100
    num_epochs = 5000
    N, data_dim = Y.shape
    hidden_dim = data_dim
    latent_dim = data_dim
    n_class = len(torch.unique(labels))
    
    # Model
    model = VAE(data_dim, hidden_dim, latent_dim, power=POWER)
    
    # split data
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(np.arange(len(Y)), test_size=0.1, random_state=2024)
    n_train, n_test = len(train_idx), len(test_idx)
    Y[train_idx] -= Y[train_idx].mean(0); Y[train_idx] /= Y[train_idx].std(0)
    Y[test_idx] -= Y[train_idx].mean(0); Y[test_idx] /= Y[train_idx].std(0)
    
    log_predprob = torch.zeros((n_class, n_test))
    loss = np.zeros((n_class, 2))
    time_ = 0
    for k in torch.unique(labels):
        # model based on training data
        Y_train = Y[train_idx][labels[train_idx]==k]
        t_train = labels[train_idx][labels[train_idx]==k]
        train_loader = DataLoader(TensorDataset(Y_train, t_train), batch_size=batch_size, shuffle=True)
        # model = VAE(data_dim, hidden_dim, latent_dim, power=POWER)
        beginning=timeit.default_timer()
        model_train, loss[k][0] = train_VAE(train_loader, model=model, num_epochs=num_epochs, 
                                                  save_model=True, class_=str(k.cpu().item())+'train', seed=seed)
        time_ += timeit.default_timer()-beginning
        elbo_train = -loss[k][0]
        # full model
        Y_full = torch.concatenate((Y_train, Y[test_idx])) # appending testing data
        t_full = torch.concatenate((t_train, labels[test_idx]))
        train_loader = DataLoader(TensorDataset(Y_full, t_full), batch_size=batch_size, shuffle=True)
        # model = VAE(data_dim, hidden_dim, latent_dim, power=POWER)
        beginning=timeit.default_timer()
        model_full, loss[k][1] = train_VAE(train_loader, model, num_epochs=num_epochs, 
                                                 save_model=True, class_=str(k.cpu().item())+'full', seed=seed)
        time_ += timeit.default_timer()-beginning
        elbo_full = -loss[k][1]
        
        # log prediction probabilities
        log_predprob[k] = elbo_full - elbo_train + torch.log(labels[test_idx].eq(k).float().mean())
    # summarize losses
    import pandas as pd
    loss = pd.DataFrame(loss, columns=['training', 'full'])
    print('Loss:\n'+loss.to_string())
    # make prediction
    from sklearn.metrics import roc_auc_score, adjusted_rand_score
    LPP = log_predprob.mean().cpu().item()
    cls_max, pred = torch.max(log_predprob, 0)
    ACC = pred.eq(labels[test_idx].view_as(pred)).float().mean().cpu()
    t_score = torch.exp(log_predprob - cls_max).T if n_class>2 else cls_max
    if t_score.size(1)>1: t_score/=t_score.sum(1,keepdims=True)
    if n_class>2:
        AUC = roc_auc_score(labels[test_idx].cpu(), t_score.detach().cpu(), multi_class='ovo')
    else:
        AUC = roc_auc_score(labels[test_idx].cpu(), t_score.detach().cpu())
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ARI = adjusted_rand_score(labels[test_idx].cpu(), pred)
    NMI = normalized_mutual_info_score(labels[test_idx].cpu(), pred)
    print('Test Accuracy: {}%'.format(100. * ACC))
    print('Test AUC: {}'.format(AUC))
    print('Test ARI: {}'.format(ARI))
    print('Test NMI: {}'.format(NMI))
    print('Test LPP: {}'.format(LPP))
    
    # save to file
    os.makedirs('./results', exist_ok=True)
    stats = np.array([ACC, AUC, ARI, NMI, LPP, time_])
    stats = np.array([seed,'q='+str(q)]+[np.array2string(r, precision=4) for r in stats])[None,:]
    header = ['seed', 'Method', 'ACC', 'AUC', 'ARI', 'NMI', 'LPP', 'time']
    f_name = os.path.join('./results','oilflow_latdim_VAE.txt')
    with open(f_name,'ab') as f:
        np.savetxt(f,stats,fmt="%s",delimiter=',',header=','.join(header) if seed==2024 else '')

if __name__ == '__main__':
    # main()
    n_seed = 10; i=0; n_success=0; n_failure=0
    while n_success < n_seed and n_failure < 10* n_seed:
        seed_i=2024+i*10
        try:
            print("Running for seed %d ...\n"% (seed_i))
            main(seed=seed_i)
            n_success+=1
        except Exception as e:
            print(e)
            n_failure+=1
            pass
        i+=1