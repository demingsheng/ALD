import time
import os
import torch
import json
import copy
import torchtuples
from utils import *
from models import *
from datasets import *
from hyperparams import *
from itertools import product
from pdb import set_trace as bb
from sklearn.model_selection import train_test_split


start_time = time.time()
# some defaults
n_runs = 5
use_gpu = True # try to use GPU (if available)
is_verbose = True # whether to print stuff out
my_rand_seed = 113

# load gpu or cpu
os.environ['CUDA_VISIBLE_DEVICES']='1'
if torch.cuda.is_available() and use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('using device:', device)

# load the datasets
datasets_str_list = [
    # Fully synthetic 1D
    'Gaussian_linear', 'Gaussian_nonlinear', 'Exponential', 'Weibull', 'LogNorm', 'Gaussian_uniform',
    
    # Fully synthetic 4D and 8D
    'Norm_med', 'Norm_heavy', 'Norm_light', 'Norm_same',
    'LogNorm_med', 'LogNorm_heavy', 'LogNorm_light', 'LogNorm_same',

    # Real datasets
    'METABRIC', 'WHAS', 'SUPPORT', 'GBSG', 'TMBImmuno', 'BreastMSK', 'LGGGBM'
]

models_str_list = [
    # Parametric & NonParametric ALD Models
    'ALD', 'CQRNN', 
    # # (Semi-)Parametric Models & Mixture Models
    'LogNorm', 'DeepSurv', 'DSM(Weibull)', 'DSM(LogNorm)'
    # # NonParametric Models
    'DeepHit', 'RSF', 'GBM'
]

eps_list = [200]
bs_list = [128]
n_hidden_list = [32]
dr_list = [0.1]

params_grid = product(eps_list, bs_list, n_hidden_list, dr_list, datasets_str_list)

for epochs, batch_size, n_hidden, dropout_rate, dataset_str in params_grid:
    model_res = {f"{model_str}": [] for model_str in models_str_list}
    for model_str in models_str_list:
        for run in range(n_runs):
            print(f"\n [Run{run+1}]: {dataset_str} - {model_str}")

            hyp = get_hyperparams(dataset_str, model_str)
            n_data = hyp['n_data']
            n_test = hyp['n_test']
            learning_rate = hyp['learning_rate']
            weight_decay = hyp['weight_decay']
            test_propotion = hyp['test_propotion']
            n_quantiles = hyp['n_quantiles']

            rand_in = run + my_rand_seed
            np.random.seed(rand_in)
            torch.manual_seed(rand_in)
            mydataset = get_dataset(dataset_str)

            # load the data from the dataset
            if mydataset.synth_target == True and mydataset.synth_censor == True:
                x_train, tte_train, cen_train, y_train, cen_indicator_train, obs_indicator_train = generate_data_synthtarget_synthcen(n_data, mydataset, x_range=[0,2], is_censor=True)
                x_test, tte_test, cen_test, y_test, cen_indicator_test, obs_indicator_test = generate_data_synthtarget_synthcen(n_test, mydataset, x_range=[0,2], is_censor=True)

            elif mydataset.synth_target == False and mydataset.synth_censor == False:
                data_train, data_test = generate_data_realtarget_realcen(mydataset, test_propotion, rand_in)
                x_train, tte_train, cen_train, y_train, cen_indicator_train, obs_indicator_train = data_train
                x_test, tte_test, cen_test, y_test, cen_indicator_test, obs_indicator_test = data_test

            x_val_torch, tte_val_torch, cen_val_torch, y_val_torch, cen_indicator_val_torch = None, None, None, None, None
            if model_str in ['DeepSurv', 'DeepHit']:
            # if model_str in ['ALD', 'DeepSurv', 'DeepHit']:
                x_train, x_val, tte_train, cen_train, tte_val, cen_val, y_train, y_val, cen_indicator_train, cen_indicator_val = train_test_split(x_train, tte_train, cen_train, y_train, cen_indicator_train, test_size=0.2, random_state=rand_in)
                x_val_torch, tte_val_torch, cen_val_torch, y_val_torch, cen_indicator_val_torch = get_torch_data(x_val, tte_val, cen_val, y_val, cen_indicator_val, device)

            x_train_torch, tte_train_torch, cen_train_torch, y_train_torch, cen_indicator_train_torch = get_torch_data(x_train, tte_train, cen_train, y_train, cen_indicator_train, device)
            x_test_torch, tte_test_torch, cen_test_torch, y_test_torch, cen_indicator_test_torch = get_torch_data(x_test, tte_test, cen_test, y_test, cen_indicator_test, device)  
        
            if is_verbose == 1:
                print(f'Proportion censored: {round(cen_indicator_train.mean(), 3)}')
                # plot_dataset(dataset_str, y_train, y_test, tte_train=tte_train, cen_train=cen_train, cen_indicator_train=cen_indicator_train, tte_test=tte_test, cen_test=cen_test, cen_indicator_test=cen_indicator_test)
                # save_dataset(dataset_str, run, x_train=x_train, tte_train=tte_train, cen_train=cen_train, y_train=y_train, cen_indicator_train=cen_indicator_train, x_test=x_test, tte_test=tte_test, cen_test=cen_test, y_test=y_test, cen_indicator_test=cen_indicator_test, save=True)

            dataset_dim = mydataset.input_dim
            y_max = round(1.2*y_train.max(), 4)  
            model, model2 = get_models(model_str, dataset_dim, n_hidden, n_quantiles = 10, is_dropout = True, dropout_rate=dropout_rate)

            loss = train_model(model_str, model, model2, dataset_str, x_train_torch, y_train_torch, cen_indicator_train_torch, x_test_torch, y_test_torch, cen_indicator_test_torch, y_max, learning_rate, weight_decay, epochs, batch_size, device, is_synth=mydataset.synth_censor)
            metrics = evaluate(model_str, model, model2, dataset_str, x_test_torch, tte_test_torch, y_test_torch, cen_indicator_test_torch, x_train_torch, y_train_torch, cen_indicator_train_torch, x_val_torch, y_val_torch, cen_indicator_val_torch, run+1, is_synth=mydataset.synth_censor)
                    
            print(model_str, metrics)
            model_res[model_str].append(metrics)

        aggregate(model_str, dataset_str, model_res[model_str])

end_time = time.time()
print('\ntime (secs) taken:',round(end_time-start_time, 3),'\n')
