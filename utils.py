import os
import math
import json
import numpy as np
import seaborn as sns
import scipy.special as sps
import torchtuples as tt
import matplotlib.pyplot as plt
from models import *
from sksurv.util import Surv
from lifelines import KaplanMeierFitter
from pycox.models import CoxPH, DeepHitSingle
from sksurv.nonparametric import kaplan_meier_estimator
from auton_survival.models.dsm import DeepSurvivalMachines
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.ensemble import RandomSurvivalForest, GradientBoostingSurvivalAnalysis
from scipy.stats import norm, expon, weibull_min, lognorm, uniform, wasserstein_distance


class StepFunction:
    def __init__(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def __call__(self, t):
        t = np.asarray(t)
        idx = np.searchsorted(self.x, t, side='right') - 1
        idx = np.clip(idx, 0, len(self.y) - 1)
        return self.y[idx]

def lr_lambda(current_step):
    warmup_steps = 20
    total_steps = 100
    if current_step < warmup_steps:
        return float(current_step) / float(warmup_steps)
    else:
        progress = float(current_step - warmup_steps) / float(total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

def plot_dataset(dataset_str, y_train, y_test, tte_train=None, cen_train=None, cen_indicator_train=None, tte_test=None, cen_test=None, cen_indicator_test=None):
    """
    Plots distributions of training and testing data (target variable, time-to-event, censoring, and event indicators).

    Parameters:
        y_train, y_test (ndarray): Target variables for train/test sets.
        dataset_str (str): Dataset name (used in title and filename).
        tte_train, tte_test (ndarray, optional): Time-to-event data.
        cen_train, cen_test (ndarray, optional): Censoring times.
        cen_indicator_train, cem_indicator_test (ndarray, optional): Censoring indicators (0 = event, 1 = censored).
    """
    # Create output directory if it doesn't exist
    save_dir = './figures/datasets'
    os.makedirs(save_dir, exist_ok=True)

    # Decide subplot layout
    if tte_train is None and cen_train is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes = axes.flatten()

    # --- Plot Training Set ---
    sns.histplot(y_train, bins=50, kde=True, ax=axes[0])
    axes[0].set(xlabel="y_train", title=f"{dataset_str}, mean: {round(np.mean(y_train), 2)}")

    if tte_train is not None:
        sns.histplot(tte_train, bins=50, kde=True, ax=axes[1])
        axes[1].set(xlabel="tte_train", title=f"{dataset_str}, mean: {round(np.mean(tte_train), 2)}")

    if cen_train is not None:
        sns.histplot(cen_train, bins=50, kde=True, ax=axes[2])
        axes[2].set(xlabel="cen_train", title=f"{dataset_str}, mean: {round(np.mean(cen_train), 2)}")

    if cen_indicator_train is not None:
        cen = cen_indicator_train.flatten().astype(bool)
        sns.histplot(y_train[~cen], bins=50, kde=True, ax=axes[3])
        axes[3].set(xlabel="y_train[obs]", title=f"{dataset_str}, mean: {round(np.mean(y_train[~cen]), 2)}")

        sns.histplot(y_train[cen], bins=50, kde=True, ax=axes[4])
        axes[4].set(xlabel="y_train[cen]", title=f"{dataset_str}, mean: {round(np.mean(y_train[cen]), 2)}")

    # --- Plot Test Set ---
    base_idx = 1 if tte_train is None else 5

    sns.histplot(y_test, bins=50, kde=True, ax=axes[base_idx])
    axes[base_idx].set(xlabel="y_test", title=f"{dataset_str}, mean: {round(np.mean(y_test), 2)}")

    if tte_test is not None:
        sns.histplot(tte_test, bins=50, kde=True, ax=axes[base_idx + 1])
        axes[base_idx + 1].set(xlabel="tte_test", title=f"{dataset_str}, mean: {round(np.mean(tte_test), 2)}")

    if cen_test is not None:
        sns.histplot(cen_test, bins=50, kde=True, ax=axes[base_idx + 2])
        axes[base_idx + 2].set(xlabel="cen_test", title=f"{dataset_str}, mean: {round(np.mean(cen_test), 2)}")

    if cen_indicator_test is not None:
        cen = cen_indicator_test.flatten().astype(bool)
        sns.histplot(y_test[~cen], bins=50, kde=True, ax=axes[base_idx + 3])
        axes[base_idx + 3].set(xlabel="y_test[obs]", title=f"{dataset_str}, mean: {round(np.mean(y_test[~cen]), 2)}")

        sns.histplot(y_test[cen], bins=50, kde=True, ax=axes[base_idx + 4])
        axes[base_idx + 4].set(xlabel="y_test[cen]", title=f"{dataset_str}, mean: {round(np.mean(y_test[cen]), 2)}")

    # Save and close
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{dataset_str}.png"))
    plt.close(fig)


def save_dataset(dataset_str, run, x_train, tte_train, cen_train, y_train, cen_indicator_train, x_test, tte_test, cen_test, y_test, cen_indicator_test, save=True):
    """
    Save training and testing dataset to .npz files under a structured directory.

    Parameters:
        dataset_str (str): Dataset name used for folder naming.
        run (int or str): Run index to distinguish different experiments.
        x_train, x_test: Feature arrays.
        tte_train, tte_test: Time-to-event arrays.
        cen_train, cen_test: Censoring time arrays.
        y_train, y_test: Target variables (e.g., min(tte, cen)).
        cen_indicator_train, cen_indicator_test: 1 = observed, 0 = censored.
        save (bool): Whether to actually save the data. Default: True.
    """
    if not save:
        return

    save_dir = f"datasets/{dataset_str}_run{run+1}"
    os.makedirs(save_dir, exist_ok=True)

    # Save training data
    train_path = os.path.join(save_dir, "train_data.npz")
    np.savez(
        train_path,
        x_train=x_train,
        tte_train=tte_train,
        cen_train=cen_train,
        y_train=y_train,
        cen_indicator_train=cen_indicator_train
    )

    # Save testing data
    test_path = os.path.join(save_dir, "test_data.npz")
    np.savez(
        test_path,
        x_test=x_test,
        tte_test=tte_test,
        cen_test=cen_test,
        y_test=y_test,
        cen_indicator_test=cen_indicator_test
    )


def get_ald_cdf(y, theta, sigma, kappa):
    """
    Compute the CDF of the Asymmetric Laplace Distribution (ALD).
    Supports both NumPy and PyTorch inputs. The return type matches the input type.
    
    Parameters:
        y, theta, sigma, kappa: All must be either numpy arrays or torch tensors.
        eps: minimum value to clip the CDF output (for numerical safety).
        max_exp: maximum value inside exp() to prevent overflow.
    
    Returns:
        CDF values in the same type as the inputs.
    """
    sqrt_2 = math.sqrt(2)

    if isinstance(y, torch.Tensor):
        # ===== PyTorch version =====
        sqrt_2 = torch.tensor(sqrt_2, device=y.device)
        cdf_1 = 1 - 1 / (1 + kappa**2) * torch.exp(-sqrt_2 * kappa * (y - theta) / sigma)
        value_to_exp = - sqrt_2 * (theta - y) / (sigma*kappa)
        safe_exp_value = torch.clamp(value_to_exp, max=80)
        cdf_2 = kappa**2 / (1 + kappa**2) * torch.exp(safe_exp_value)
        cdf = torch.where(y > theta, cdf_1, cdf_2)
        return cdf

    elif isinstance(y, np.ndarray):
        # ===== NumPy version =====
        value_to_exp = -sqrt_2 * (theta - y) / (sigma * kappa)
        safe_exp_value = np.clip(value_to_exp, a_min=None, a_max=80)
        cdf_1 = 1 - 1 / (1 + kappa**2) * np.exp(-sqrt_2 * kappa * (y - theta) / sigma)
        cdf_2 = kappa**2 / (1 + kappa**2) * np.exp(safe_exp_value)
        cdf = np.where(y > theta, cdf_1, cdf_2)
        return cdf

    else:
        raise TypeError("Inputs must be either torch.Tensor or np.ndarray (and all of the same type).")


def get_cqrnn_cdf(quantiles, test_y):
    """
    Compute the approximate CDF values for test_y based on the predicted quantiles.
    Args:
        quantiles (np.ndarray): Array of predicted quantiles with shape (1000, 100),
                                where each row contains 100 quantile values for a sample.
        test_y (np.ndarray): Test target values with shape (1000, n),
                             where each column is a different sample to evaluate.
    Returns:
        np.ndarray: Array of shape (1000, n), where each element represents the estimated CDF
                    probability (i.e., the quantile level) corresponding to the nearest quantile
                    in the predicted quantile set for that sample.
    """
    num_samples, num_quantiles = quantiles.shape
    _, num_test = test_y.shape

    # Generate quantile levels, e.g., [0.0, 0.01, ..., 1.0]
    probabilities = np.linspace(0, 1, num_quantiles)

    # Compute index of the closest quantile for each test value
    indices = np.abs(quantiles[:, :, np.newaxis] - test_y[:, np.newaxis, :]).argmin(axis=1)

    # Use these indices to retrieve the corresponding quantile probabilities
    cdf = probabilities[indices]

    return cdf


def get_models(model_str, dataset_dim, hidden_dim, q_dim = 4, n_quantiles = 10, is_positive = False, is_dropout = True, dropout_rate = 0.1):
    """
    Factory for instantiating primary and secondary models.
    Parameters:
        model_str: One of
            'ALD', 'CQRNN', 'Pre_ALD_Cal', 'Pre_ALD_Cqr', 'Post_ALD_Cal', 'Post_ALD_Cqr', 'LogNorm'
            
        dataset_dim (int): Dimensionality of input features.
        hidden_dim (int): Number of hidden units.
        q_dim (int): Number of quantile outputs for calibration models.
        n_quantiles (int): Number of quantiles for CQRNN.
        use_batch_norm (bool): Whether to include BatchNorm layers.
        use_dropout (bool): Whether to include Dropout layers.
        dropout_rate (float): Dropout probability if use_dropout is True.
    Returns:
        model:      Primary nn.Module instance.
        model2:     Secondary network for post-calibration
    """
    model2 = MLP_Base(input_dim=dataset_dim+1, n_hidden=16, output_dim=3, is_positive=True, is_dropout=False, dropout_rate=None)
    if model_str in ['ALD', 'Post_ALD_Cal', 'Post_ALD_Cqr']:
        # model = MLP_ALD(input_dim=dataset_dim, n_hidden=hidden_dim, output_dim=1, is_dropout=is_dropout, dropout_rate=dropout_rate)
        model = MLP_ALD(input_dim=dataset_dim, n_hidden=hidden_dim, output_dim=1, is_dropout=False, dropout_rate=None)

    elif model_str == 'CQRNN':
        # model = MLP_Base(input_dim=dataset_dim, n_hidden=hidden_dim, output_dim=n_quantiles, is_dropout=is_dropout, dropout_rate=dropout_rate)
        model = MLP_Base(input_dim=dataset_dim, n_hidden=hidden_dim, output_dim=100, is_dropout=is_dropout, dropout_rate=dropout_rate)

    elif model_str ==  "LogNorm":
        model = MLP_Base(input_dim=dataset_dim, n_hidden=hidden_dim, output_dim=2, is_dropout=is_dropout, dropout_rate=dropout_rate)

    else:
        model = None
        
    return model, model2


def get_torch_data(x, event_time, cen_time, y, cen_indicator, device):
    """
    Convert NumPy or array-like survival data to torch Tensors on the given device.
    Parameters:
        x (array-like): Feature matrix, shape (N, D).
        event_time (array-like): Time-to-event array, shape (N,1).
        cen_time (array-like): Censoring times, shape (N,1).
        y (array-like): Observed times (min(event_time, cen_time)), shape (N,1).
        cen_indicator (array-like): Event indicator (0=event, 1=censored), shape (N,1).
        device (torch.device or str): Target device for tensors.
    Returns:
        x_torch            (Tensor[N, D], float32)
        event_time_torch   (Tensor[N, 1], float32)
        cen_time_torch     (Tensor[N, 1], float32)
        y_torch            (Tensor[N, 1], float32)
        cen_indicator_torch(Tensor[N, 1], float32)
    """
    # Features
    x_torch = torch.as_tensor(x, dtype=torch.float32, device=device)

    # Times and indicators
    event_time_torch    = torch.as_tensor(event_time,    dtype=torch.float32, device=device)
    cen_time_torch      = torch.as_tensor(cen_time,      dtype=torch.float32, device=device)
    y_torch             = torch.as_tensor(y,             dtype=torch.float32, device=device)
    cen_indicator_torch = torch.as_tensor(cen_indicator, dtype=torch.float32, device=device)

    return x_torch, event_time_torch, cen_time_torch, y_torch, cen_indicator_torch



def train_model(model_str, model, model2, dataset_str, x_train_torch, y_train_torch, cen_indicator_train_torch, x_test_torch, y_test_torch, cen_indicator_test_torch, y_max, learning_rate, weight_decay, epochs, batch_size, device, is_synth=True, x_val_torch=None, y_val_torch=None, cen_indicator_val_torch=None, patience=10, weight=0.1, is_verbose=1):
    """
    Train `model` (optionally with `model2` for post-calibration) on training data,
    then evaluate on test data. ALD-based models support optional validation with early stopping.
    """
    valid_models = ['ALD', 'CQRNN', 'LogNorm']
    if model_str not in valid_models:
        return None

    model.to(device)
    if model2:
        model2.to(device)

    # Initialize logs
    logs = {
        'train': ([], [], []),  # loss, nll, cal
        'test':  ([], [], []),
        'val':   ([], [], []),
        'post':  ([], [])
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_val_loss, stop_counter = float('inf'), 0
    is_lognorm = dataset_str in ['LogNorm_heavy', 'LogNorm_med', 'LogNorm_light', 'LogNorm_same']

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(x_train_torch.size(0), device=device)
        train_loss_ep, nll_ep, cal_ep = 0.0, 0.0, 0.0

        for i in range(0, x_train_torch.size(0), batch_size):
            idx = perm[i:i + batch_size]
            x_b, y_b, cen_b = x_train_torch[idx], y_train_torch[idx], cen_indicator_train_torch[idx]

            if model_str == 'ALD':
                theta, sigma, kappa = model(x_b.to(device))
                loss = loss_ald(y_b, theta, sigma, kappa, cen_b)
                loss_nll = loss_cal = None

            elif model_str == 'CQRNN':
                y_pred = model(x_b.to(device))
                loss = loss_cqr(y_b, y_pred, y_max, cen_b)
                loss_nll = loss_cal = None

            elif model_str == 'LogNorm':
                y_pred = model(x_b.to(device))
                loss = loss_lognorm(y_b, y_pred, cen_b)
                loss_nll = loss_cal = None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss_ep += loss.item()
            if loss_nll is not None:
                nll_ep += loss_nll.item()
            if loss_cal is not None:
                cal_ep += loss_cal.item()

        n_train = x_train_torch.size(0)
        logs['train'][0].append(round(train_loss_ep / n_train, 4))
        logs['train'][1].append(round(nll_ep / n_train, 4) if nll_ep else 0.0)
        logs['train'][2].append(round(cal_ep / n_train, 4) if cal_ep else 0.0)

        # Evaluation
        model.eval()
        test_loss_ep = 0.0
        with torch.no_grad():
            for j in range(0, x_test_torch.size(0), batch_size):
                x_b = x_test_torch[j:j + batch_size]
                y_b = y_test_torch[j:j + batch_size]
                cen_b = cen_indicator_test_torch[j:j + batch_size]

                if model_str in ['ALD', 'Post_ALD_Cal', 'Post_ALD_Cqr']:
                    theta, sigma, kappa = model(x_b.to(device))
                    loss = loss_ald(y_b, theta, sigma, kappa, cen_b)

                elif model_str in ['Pre_ALD_Cal', 'Pre_ALD_Cqr']:
                    q = torch.rand(x_b.size(0), 1, device=device)
                    theta, sigma, kappa = model(x_b, q)
                    if model_str == 'Pre_ALD_Cal':
                        loss, _, _ = loss_ald_cal(y_b, theta, sigma, kappa, cen_b, q)
                    else:
                        loss, _, _ = loss_ald_cqr(y_b, y_max, theta, sigma, kappa, cen_b, q)

                elif model_str == 'CQRNN':
                    y_pred = model(x_b.to(device))
                    loss = loss_cqr(y_b, y_pred, y_max, cen_b)

                elif model_str == 'LogNorm':
                    y_pred = model(x_b.to(device))
                    loss = loss_lognorm(y_b, y_pred, cen_b)

                test_loss_ep += loss.item()

        logs['test'][0].append(round(test_loss_ep / x_test_torch.size(0), 4))
        logs['test'][1].append(0.0)
        logs['test'][2].append(0.0)

        # Validation with early stopping (ALD-based)
        # if model_str == 'ALD':
        #     val_loss_ep = 0.0
        #     with torch.no_grad():
        #         for i in range(0, x_val_torch.size(0), batch_size):
        #             x_b = x_val_torch[i: i + batch_size]
        #             y_b = y_val_torch[i: i + batch_size]
        #             cen_b = cen_indicator_val_torch[i: i + batch_size]

        #             theta, sigma, kappa = model(x_b.to(device))
        #             loss = loss_ald(y_b, theta, sigma, kappa, cen_b)
        #             val_loss_ep += loss.item()

        #     n_val = x_val_torch.size(0)
        #     avg_val_loss = round(val_loss_ep / n_val, 4)
        #     val_losses.append(avg_val_loss)
        #     val_NLL_losses.append(0.0)
        #     val_Cal_losses.append(0.0)

        #     # Early stop check
        #     if avg_val_loss < best_val_loss:
        #         best_val_loss = avg_val_loss
        #         stop_counter = 0
        #     else:
        #         stop_counter += 1
        #     if stop_counter >= patience:
        #         if is_verbose:
        #             print(f"Early stopping at epoch {ep+1}: no improvement in {patience} epochs.")
        #         break

        if is_verbose:
            print(f"[Epoch {ep + 1}/{epochs}] Train Loss: {logs['train'][0][-1]}", end='\r')

    return logs

def evaluate(model_str, model, model2, dataset_str, x_test_torch, tte_test_torch, y_test_torch, cen_indicator_test_torch, x_train_torch, y_train_torch, cen_indicator_train_torch, x_val_torch, y_val_torch, cen_indicator_val_torch, n_runs, is_synth=True):

    time_grid = np.linspace(np.percentile(y_train_torch.reshape(-1).detach().numpy(), 10), np.percentile(y_train_torch.reshape(-1).detach().numpy(), 90), 100)
    q = torch.cat((torch.linspace(0.05, 0.95, steps=19), torch.tensor([0.9999])))
    with torch.no_grad():
        if model_str == "ALD":
            theta, sigma, kappa = model(x_test_torch)
            y_test_pred = (theta + sigma/np.sqrt(2) * (1/kappa - kappa)).detach().numpy()
            quantiles = get_quantiles(model_str, ald_params={'theta': theta, 'sigma': sigma, 'kappa': kappa}, q=q)
            ald_cdfs = get_ald_cdf(y_test_torch, theta, sigma, kappa).detach().numpy()
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, ald_params={'theta': theta.detach().numpy(), 'sigma': sigma.detach().numpy(), 'kappa': kappa.detach().numpy()})
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=ald_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=ald_cdfs)
            
        if model_str == "CQRNN":
            quantiles = model(x_test_torch)
            y_test_pred = quantiles[:,quantiles.shape[1]//2 - 1].detach().numpy().reshape(-1, 1)
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid)
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 9::10].detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy())
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles[:, 9::10].detach().numpy())
            
        if model_str == 'LogNorm':
            lognorm_pramas = model(x_test_torch)
            lognorm_mean = lognorm_pramas[:,0:1].detach().numpy()
            soft_fn = nn.Softplus()
            lognorm_sd = soft_fn(lognorm_pramas[:,1:2]).detach().numpy()
            y_test_pred = np.exp(lognorm_mean+lognorm_sd**2/2).reshape(-1, 1)
            lognorm_cdfs = lognorm.cdf(y_test_torch.detach().numpy(), s=lognorm_sd, scale=np.exp(lognorm_mean))
            quantiles = get_quantiles(model_str, lognorm_params={'mean': lognorm_mean, 'sd': lognorm_sd}, q=q)
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, lognorm_params={'mean': lognorm_mean, 'sd': lognorm_sd})
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=lognorm_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=lognorm_cdfs)
            
        if model_str == "DeepSurv":
            net = tt.practical.MLPVanilla(x_train_torch.shape[1], [32,32], 1, True, 0.1)
            model = CoxPH(net, tt.optim.Adam)
            model.optimizer.set_lr(0.01)
            target = (y_train_torch.view(-1), 1 - cen_indicator_train_torch.view(-1))
            target_val = (y_val_torch.view(-1), 1 - cen_indicator_val_torch.view(-1))
            val = x_val_torch, target_val
            callbacks = [tt.callbacks.EarlyStopping()]
            train_loss = model.fit(x_train_torch, target, 256, 200, callbacks, val_data=val, verbose=False)
            _ = model.compute_baseline_hazards()
            surv = model.predict_surv_df(x_test_torch)
            y_test_pred = get_median_durations(model_str, surv, y_median = np.median(y_train_torch.reshape(-1).detach().numpy())).reshape(-1, 1)
            quantiles = get_quantiles(model_str, cdf=surv, q=q)
            durations = surv.index.to_numpy()  
            idx = np.abs(durations[:, None] - y_test_torch.reshape(-1).detach().numpy()[None, :]).argmin(axis=0)
            deepsurv_cdfs = 1 - surv.values[idx, np.arange(surv.shape[1])].reshape(-1, 1)
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, cdf=surv)
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=deepsurv_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=deepsurv_cdfs)
            
        if model_str == "DeepHit":
            num_durations = 100
            labtrans = DeepHitSingle.label_transform(num_durations)
            get_target = lambda _: (y_train_torch.view(-1).numpy(), 1 - cen_indicator_train_torch.view(-1).numpy())
            target = labtrans.fit_transform(*get_target(None))
            target = (torch.from_numpy(target[0]).long(), torch.from_numpy(target[1]).long())
            get_target_val = lambda _: (y_val_torch.view(-1).numpy(), 1 - cen_indicator_val_torch.view(-1).numpy())
            target_val = labtrans.fit_transform(*get_target_val(None))
            target_val = (torch.from_numpy(target_val[0]).long(), torch.from_numpy(target_val[1]).long())
            val = (x_val_torch, target_val)
            net = tt.practical.MLPVanilla(x_train_torch.shape[1], [32, 32], num_durations, True, 0.1)
            model = DeepHitSingle(net, tt.optim.Adam, alpha=0.2, sigma=0.1, duration_index=labtrans.cuts)
            model.optimizer.set_lr(0.01)
            callbacks = [tt.callbacks.EarlyStopping()]
            train_loss = model.fit(x_train_torch, target, 256, 200, callbacks, val_data=val, verbose=False)
            surv = model.predict_surv_df(x_test_torch)
            y_test_pred = get_median_durations(model_str, surv, y_train=y_train_torch.reshape(-1).detach().numpy(), num_durations=num_durations).reshape(-1, 1)
            quantiles = get_quantiles(model_str, cdf=surv, q=q)
            durations = surv.index.to_numpy()  
            idx = np.abs(durations[:, None] - y_test_torch.reshape(-1).detach().numpy()[None, :]).argmin(axis=0)
            deephit_cdfs = 1 - surv.values[idx, np.arange(surv.shape[1])].reshape(-1, 1)
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, cdf=surv)
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=deephit_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=deephit_cdfs)
            
        if model_str == 'GBM':
            model = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=n_runs)
            structured_y_train = np.array([(bool(e), t) for e, t in zip(1 - cen_indicator_train_torch.reshape(-1).detach().numpy(), y_train_torch.reshape(-1).detach().numpy())], dtype=[('event', '?'), ('time', '<f8')])
            model.fit(x_train_torch.detach().numpy(), structured_y_train)
            surv = model.predict_survival_function(x_test_torch.detach().numpy())
            y_test_pred = np.array([get_median_survival_time(sf) for sf in surv]).reshape(-1, 1)
            quantiles = get_quantiles(model_str, cdf=surv, q=q)
            gbm_cdfs = np.array([1.0 if t > sf.x[-1] else 1 - sf(t) for sf, t in zip(surv, y_test_torch.reshape(-1).detach().numpy())])
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, cdf=surv)
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=gbm_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=gbm_cdfs)
            
        if model_str == 'RSF':
            model = RandomSurvivalForest(100, random_state=n_runs)
            structured_y_train = np.array([(bool(e), t) for e, t in zip(1 - cen_indicator_train_torch.reshape(-1).detach().numpy(), y_train_torch.reshape(-1).detach().numpy())], dtype=[('event', '?'), ('time', '<f8')])
            model.fit(x_train_torch.detach().numpy(), structured_y_train)
            surv = model.predict_survival_function(x_test_torch.detach().numpy())
            y_test_pred = np.array([get_median_survival_time(sf) for sf in surv])
            quantiles = get_quantiles(model_str, cdf=surv, q=q)
            rsf_cdfs = np.array([1.0 if t > sf.x[-1] else 1 - sf(t) for sf, t in zip(surv, y_test_torch.reshape(-1).detach().numpy())])
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, cdf=surv)
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=rsf_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=rsf_cdfs)
            
        if model_str in ['DSM(Weibull)', 'DSM(LogNorm)']:
            if model_str == 'DSM(Weibull)':
                model = DeepSurvivalMachines(k=100, distribution='Weibull', layers=[32, 32])
            else:
                model = DeepSurvivalMachines(k=10, distribution='LogNormal', layers=[32, 32])
            
            x_train = x_train_torch.detach().numpy().astype(np.float64)
            x_test = x_test_torch.detach().numpy().astype(np.float64)
            model.fit(x_train, y_train_torch.detach().numpy(), 1 - cen_indicator_train_torch.reshape(-1).detach().numpy())
            time_grid = np.linspace(0, y_train_torch.max(), 1000)
            surv = [] 
            for t in time_grid:
                surv_t = model.predict_survival(x_test, t)
                surv.append(surv_t.reshape(-1, 1))
            surv = np.hstack(surv)
            cdf = []
            for i in range(surv.shape[0]):
                cdf_vals = 1 - surv[i]  # CDF = 1 - S(t)
                step_fn = StepFunction(time_grid, cdf_vals)
                cdf.append(step_fn)
            y_test_pred = np.array([get_median_dsm_survival_time(item) for item in cdf]).reshape(-1, 1)
            quantiles = get_quantiles(model_str, cdf=cdf, q=q)
            dsm_cdfs = np.array([0 if t > cdf_vals.x[-1] else cdf_vals(t) for cdf_vals, t in zip(cdf, y_test_torch.reshape(-1).detach().numpy())])
            ibs = calculate_ibs(model_str, y_test_torch.detach().numpy(), 1 - cen_indicator_test_torch.detach().numpy(), y_test_pred, time_grid, cdf=cdf)
            dcal_cens = calculate_dcal(model_str, y_test_torch.reshape(-1).detach().numpy(), quantiles[:, 1::2], cen_indicator_test_torch.reshape(-1).detach().numpy(), cdf=dsm_cdfs)
            CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept = calculate_slope_intercept(model_str, y_test_torch.reshape(-1).detach().numpy(), cen_indicator_test_torch.reshape(-1).detach().numpy(), y_test_pred=quantiles, cdf=dsm_cdfs)
            
    if is_synth == True:
        true_mean, _ = get_ground_truth(dataset_str, x_test_torch.detach().numpy())
        mae = np.mean(np.abs(y_test_pred - true_mean))
    else:
        mae = np.mean(np.abs(y_test_pred[cen_indicator_test_torch.reshape(-1).detach().numpy() == 0] - y_test_torch.detach().numpy()[cen_indicator_test_torch.reshape(-1).detach().numpy() == 0]))

    c_index_H, *_ = concordance_index_censored(cen_indicator_test_torch.reshape(-1).detach().numpy() == 0, y_test_torch.reshape(-1).detach().numpy(), -y_test_pred.reshape(-1))

    time, survival_prob = kaplan_meier_estimator(cen_indicator_train_torch.reshape(-1).detach().numpy()==0, y_train_torch.reshape(-1).detach().numpy())
    max_time = time[survival_prob > 0][-1]
    survival_train = Surv.from_arrays(cen_indicator_train_torch.reshape(-1).detach().numpy() == 0, y_train_torch.reshape(-1).detach().numpy()) 
    survival_test = Surv.from_arrays(cen_indicator_test_torch.reshape(-1).detach().numpy() == 0, y_test_torch.reshape(-1).detach().numpy())
    c_index_U, *_ = concordance_index_ipcw(survival_train, survival_test, -y_test_pred.reshape(-1), tau=max_time)

    keys = ['mae', 'ibs', 'c_index_H', 'c_index_U', 'dcal_cens', 'CalS_Slope', 'CalS_Intercept', 'Calf_Slope', 'Calf_Intercept']
    values = [mae, ibs, c_index_H, c_index_U, dcal_cens, CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept]
    metrics = {k: float(v) for k, v in zip(keys, values)}

    return metrics


def get_quantiles(model_str, ald_params=None, lognorm_params=None, cdf=None, q=torch.cat((torch.linspace(0.05, 0.95, steps=19), torch.tensor([0.9999])))):

    if model_str == "ALD":
        theta, sigma, kappa = ald_params['theta'], ald_params['sigma'], ald_params['kappa']
        quantiles_1 = theta + sigma * kappa * torch.log((1+kappa**2)*q/kappa**2)/torch.sqrt(torch.tensor([2]))
        quantiles_2 = theta - sigma * torch.log((1+kappa**2)*(1-q))/(torch.sqrt(torch.tensor([2]))*kappa)
        quantiles = torch.where(q < kappa**2/(1+kappa**2), quantiles_1, quantiles_2).detach().numpy()

    if model_str == "LogNorm":
        sd, mean = lognorm_params['sd'], lognorm_params['mean']
        quantiles = lognorm.ppf(q, sd, scale=np.exp(mean))

    if model_str in ['DeepSurv', 'DeepHit']:
        durations = cdf.index.to_numpy()  
        quantiles = []
        for q in np.linspace(0.05, 0.95, 19):
            condition = (1 - cdf.values) >= q
            idx = np.argmax(condition, axis=0)  
            idx[~condition.any(axis=0)] = len(durations) - 1  
            quantiles.append(durations[idx])
        quantiles = np.array(quantiles).T
        quantiles = np.hstack((quantiles, np.full((quantiles.shape[0], 1), durations[-1], dtype=quantiles.dtype)))
    
    if model_str in ['RSF', 'GBM']:
        quantiles = []
        q_list = np.linspace(0.05, 0.95, 19)
        for sf in cdf:
            time_x = sf.x
            time_sf = sf.y
            cdf = 1 - time_sf
            times = []
            for q in q_list:
                idx = np.where(cdf >= q)[0]
                if len(idx) == 0:
                    times.append(time_x[-1])
                else:
                    times.append(time_x[idx[0]])
            times.append(time_x[-1])
            quantiles.append(times)
        quantiles = np.array(quantiles)

    if model_str in ['DSM(Weibull)', 'DSM(LogNorm)']:
        quantiles = []
        q_list = np.linspace(0.05, 0.95, 19)
        for sf in cdf:
            time_x = sf.x
            dsm_cdf = sf.y
            times = []
            for q in q_list:
                idx = np.where(dsm_cdf >= q)[0]
                if len(idx) == 0:
                    times.append(time_x[-1])
                else:
                    times.append(time_x[idx[0]])
            times.append(time_x[-1])
            quantiles.append(times)
        quantiles = np.array(quantiles)

    return quantiles


def calculate_ibs(model_str, y_test, obs_indicator_test, y_test_pred, time_grid, ald_params=None, lognorm_params=None, cdf=None, **kwargs):
    """
    Calculate the Integrated Brier Score (IBS) for the test set.
    """
    brier_scores = []
    for time_t in time_grid:
        brier_score = calculate_bs_at_t(model_str, y_test, obs_indicator_test, y_test_pred, time_t, ald_params=ald_params, lognorm_params=lognorm_params, cdf=cdf)
        brier_scores.append(brier_score)
    # Integrate the Brier Score over the time grid
    ibs = np.trapz(brier_scores, time_grid) / (time_grid[-1] - time_grid[0])
    
    return ibs


def calculate_bs_at_t(model_str, y_test, obs_indicator_test, y_test_pred, time_t, ald_params=None, lognorm_params=None, cdf=None, **kwargs):
    """
    Calculate the Brier Score at time t for the test set.
    """
    kmf = KaplanMeierFitter()
    kmf.fit(y_test.reshape(-1), obs_indicator_test.reshape(-1))
    G_t = kmf.predict(time_t)
    G_y = kmf.predict(y_test.reshape(-1))    
    G_y = G_y[~G_y.index.duplicated(keep='first')]
    brier_scores = np.zeros(len(y_test))
    if model_str in ['ALD', 'Pre_ALD_Cal', 'Pre_ALD_Cqr', 'Post_ALD_Cal', 'Post_ALD_Cqr']:
        if model_str in ['ALD']:
            theta, sigma, kappa = ald_params['theta'], ald_params['sigma'], ald_params['kappa']
            survival_score = 1 - get_ald_cdf(np.array(time_t), theta, sigma, kappa)
        if model_str in ["Pre_ALD_Cal", "Pre_ALD_Cqr", 'Post_ALD_Cal', 'Post_ALD_Cqr']:
            survival_score_list = []
            for i in range(len(ald_params['theta'])):
                theta, sigma, kappa = ald_params['theta'][i], ald_params['sigma'][i], ald_params['kappa'][i]
                survival_score_list.append(1 - get_ald_cdf(np.array(time_t), theta, sigma, kappa))
            survival_score = np.mean(np.stack(survival_score_list, axis=0), axis=0)
    
    if model_str == "CQRNN":
        scores = 1 - np.linspace(1/y_test_pred.shape[1], 1, y_test_pred.shape[1])
        survival_score = np.zeros((len(y_test), 1))
        for i in range(y_test_pred.shape[1] - 1):  
            indices = (y_test_pred[:, i] <= time_t) & (time_t < y_test_pred[:, i+1])
            dist_to_i = np.abs(time_t - y_test_pred[indices, i])
            dist_to_i_plus_1 = np.abs(time_t - y_test_pred[indices, i + 1])
            survival_score[indices, 0] = np.where(dist_to_i <= dist_to_i_plus_1, scores[i], scores[i + 1])
        indices = time_t >= y_test_pred[:, y_test_pred.shape[1] - 1]
        survival_score[indices, 0] = 0 
    
    if model_str == 'LogNorm':
        lognorm_mean, lognorm_sd = lognorm_params['mean'], lognorm_params['sd']
        survival_score = lognorm.cdf(time_t, s=lognorm_sd, scale=np.exp(lognorm_mean))

    if model_str in ['DeepSurv', 'DeepHit']:
        durations = cdf.index.to_numpy()
        idx = (np.abs(durations - time_t)).argmin()
        survival_score = 1 - cdf.iloc[idx].to_numpy()

    if model_str in ['RSF', 'GBM']:
        survival_score = np.array([sf(time_t) for sf in cdf])
    
    if model_str in ['DSM(Weibull)', 'DSM(LogNorm)']:
        survival_score = np.array([1 - sf(time_t) for sf in cdf])
    
    for i in range(len(y_test)):
        if y_test[i] <= time_t and obs_indicator_test[i] == 1:
            brier_scores[i] = ((0 - survival_score[i])**2) / G_y[y_test[i]]
        elif y_test[i] > time_t:
            brier_scores[i] = ((1 - survival_score[i])**2) / G_t

    return np.mean(brier_scores)


def calculate_dcal(model_str, y_test, y_test_pred, cen_indicator_test, n_quantiles=10, cdf=None):
    taus = np.linspace(1/n_quantiles, 1, n_quantiles)
    # Uncensored D-Calibration (UnDCal)
    calibration_data = []
    calibration_data.append([0.0, 0.0])  # First point at (0, 0)
    
    # Compute uncensored calibration for each quantile
    for i in range(n_quantiles - 1):
        cal_prop_target = taus[i]
        cal_prop_larger = np.mean(y_test_pred[cen_indicator_test == 0, i] > y_test[cen_indicator_test == 0])
        calibration_data.append([cal_prop_target, cal_prop_larger])
    calibration_data.append([1.0, 1.0])  # Final point at (1, 1)
    calibration_data = np.array(calibration_data)
    
    # Calculate UnDCal (Difference in captured proportions and target)
    dcal_data = []
    for i in range(calibration_data.shape[0] - 1):
        target = calibration_data[i + 1, 0] - calibration_data[i, 0]
        captured = calibration_data[i + 1, 1] - calibration_data[i, 1]
        dcal_data.append([target, captured])

    dcal_data = np.array(dcal_data)
    dcal_nocens = 100 * np.sum(np.square(dcal_data[:, 0] - dcal_data[:, 1]))

    # Censored D-Calibration (CensDCal)
    diffs = y_test_pred[:, :-1] - np.expand_dims(y_test, axis=1)
    closest_q_idx = np.argmin(np.abs(diffs), axis=1)
    closest_q = np.array([taus[closest_q_idx[i]] for i in range(y_test.shape[0])])

    if model_str in ['ALD', 'LogNorm', 'DeepSurv', 'DeepHit']:
        closest_q = cdf.squeeze()
        closest_q = np.where(closest_q == 1, 0.9999, closest_q)

    dcal_data_cens = []
    for i in range(n_quantiles):
        a = taus[i-1] if i > 0 else 0.0
        b = taus[i]
        
        # Uncensored data points
        if b < 1.0:
            smaller_b = y_test[cen_indicator_test == 0] <= y_test_pred[cen_indicator_test == 0, i]
            smaller_b_cens = y_test[cen_indicator_test == 1] < y_test_pred[cen_indicator_test == 1, i]
        else:
            smaller_b = 1e9 > y_test[cen_indicator_test == 0] 
            smaller_b_cens = 1e9 > y_test[cen_indicator_test == 1] 
        
        if a > 0.0:
            larger_a = y_test[cen_indicator_test == 0] >= y_test_pred[cen_indicator_test == 0, i-1]
            larger_a_cens = y_test[cen_indicator_test == 1] >= y_test_pred[cen_indicator_test == 1, i-1]
            smaller_a_cens = y_test[cen_indicator_test == 1] < y_test_pred[cen_indicator_test == 1, i-1]
        else:
            larger_a = -1e9 <= y_test[cen_indicator_test == 0] 
            larger_a_cens = -1e9 <= y_test[cen_indicator_test == 1]
            smaller_a_cens = -1e9 > y_test[cen_indicator_test == 1]
        
        fallwithin = smaller_b * larger_a
        fallwithin_cens = smaller_b_cens * larger_a_cens
        cens_part1 = fallwithin_cens * (b - closest_q[cen_indicator_test == 1]) / (1 - closest_q[cen_indicator_test == 1])
        cens_part2 = smaller_a_cens * (b - a) / (1 - closest_q[cen_indicator_test == 1])

        total_points = fallwithin.sum() + cens_part1.sum() + cens_part2.sum()
        prop_captured = total_points / y_test.shape[0]
        dcal_data_cens.append([b - a, prop_captured])
         
    dcal_data_cens = np.array(dcal_data_cens)
    dcal_cens = 100 * np.sum(np.square(dcal_data_cens[:, 0] - dcal_data_cens[:, 1]))


    return dcal_cens


def calculate_slope_intercept(model_str, y_test, cen_indicator_test, y_test_pred=None, cdf=None):
    if model_str in ['ALD', 'LogNorm', 'DeepSurv', 'DSM(Weibull)', 'DSM(LogNorm)', 'DeepHit', 'RSF', 'GBM']:
        fcens_ratio = get_fcens_ratio(model_str, y_test, y_test_pred, cen_indicator_test, cdf=cdf)
        Calf_Slope, Calf_Intercept = np.polyfit(np.linspace(0.1, 0.9, 9), fcens_ratio, 1)
        Scens_ratio = get_Scens_ratio(model_str, y_test, y_test_pred[:, 1::2], cen_indicator_test, cdf=cdf)

    if model_str == 'CQRNN':
        fcens_ratio = get_fcens_ratio(model_str, y_test, y_test_pred, cen_indicator_test)
        Calf_Slope, Calf_Intercept = np.polyfit(np.array([0.2, 0.4, 0.6, 0.8]), fcens_ratio, 1)
        Scens_ratio = get_Scens_ratio(model_str, y_test, y_test_pred, cen_indicator_test)
        
    Scens_ratio[-1] = 1.0
    CalS_Slope, CalS_Intercept = np.polyfit(np.linspace(0.1, 1.0, 10), Scens_ratio, 1)
    
    return CalS_Slope, CalS_Intercept, Calf_Slope, Calf_Intercept


def get_fcens_ratio(model_str, y_test, y_test_pred, cen_indicator_test, cdf=None):
    
    dcal_data_cens = []
    
    if model_str in ['ALD', 'LogNorm', 'DeepSurv', 'DSM(Weibull)', 'DSM(LogNorm)', 'DeepHit', 'RSF', 'GBM']: 
        n_quantiles = 20
        taus = np.linspace(1/n_quantiles, 1, n_quantiles)
        closest_q = cdf.squeeze()
        closest_q = np.where(closest_q == 1, 0.9999, closest_q)

    if model_str == 'CQRNN':
        n_quantiles = 10
        taus = np.linspace(1/n_quantiles, 1, n_quantiles)
        diffs = y_test_pred[:, :-1] - np.expand_dims(y_test, axis=1)
        closest_q_idx = np.argmin(np.abs(diffs), axis=1)
        closest_q = np.array([taus[closest_q_idx[i]] for i in range(y_test.shape[0])])
        

    for i in range(n_quantiles):
        a = taus[i-1] if i > 0 else 0.0
        b = taus[i]
        # Uncensored data points
        if b < 1.0:
            smaller_b = y_test[cen_indicator_test == 0] <= y_test_pred[cen_indicator_test == 0, i] 
            smaller_b_cens = y_test[cen_indicator_test == 1] <= y_test_pred[cen_indicator_test == 1, i]
        else:
            smaller_b =  y_test[cen_indicator_test == 0] < 1e9
            smaller_b_cens = y_test[cen_indicator_test == 1] < 1e9
        
        if a > 0.0:
            larger_a = y_test[cen_indicator_test == 0] >= y_test_pred[cen_indicator_test == 0, i-1]
            larger_a_cens = y_test[cen_indicator_test == 1] >= y_test_pred[cen_indicator_test == 1, i-1]
            smaller_a_cens = y_test[cen_indicator_test == 1] <= y_test_pred[cen_indicator_test == 1, i-1] 
        else:
            larger_a = y_test[cen_indicator_test == 0] >= -1e9 
            larger_a_cens = y_test[cen_indicator_test == 1] >= -1e9 
            smaller_a_cens = y_test[cen_indicator_test == 1] <= -1e9
        
        fallwithin = smaller_b * larger_a
        fallwithin_cens = smaller_b_cens * larger_a_cens
        cens_part1 = fallwithin_cens * (b - closest_q[cen_indicator_test == 1]) / (1 - closest_q[cen_indicator_test == 1])
        cens_part2 = smaller_a_cens * (b - a) / (1 - closest_q[cen_indicator_test == 1])

        total_points = fallwithin.sum() + cens_part1.sum() + cens_part2.sum()
        prop_captured = total_points / y_test.shape[0]

        dcal_data_cens.append([b - a, prop_captured])
    dcal_data_cens = np.array(dcal_data_cens)
    
    fcens_ratio = []
    if model_str in ['ALD', 'LogNorm', 'DeepSurv', 'DSM(Weibull)', 'DSM(LogNorm)', 'DeepHit', 'RSF', 'GBM']:  
        for i in range(9):   
            sum_value = sum(dcal_data_cens[:,1][9-i: 11+i]) 
            fcens_ratio.append(sum_value)

    if model_str == 'CQRNN':
        for i in range(4):   
            sum_value = sum(dcal_data_cens[:,1][4-i: 6+i]) 
            fcens_ratio.append(sum_value)
  
    return fcens_ratio


def get_Scens_ratio(model_str, y_test, y_test_pred, cen_indicator_test, cdf=None):
    
    dcal_data_cens = []
    taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    n_quantiles = len(taus)

    if model_str in ['ALD', 'LogNorm', 'DeepSurv', 'DSM(Weibull)', 'DSM(LogNorm)', 'DeepHit', 'RSF', 'GBM']: 
        closest_q = cdf.squeeze()
        closest_q = np.where(closest_q == 1, 0.9999, closest_q)


    if model_str == 'CQRNN':
        diffs = y_test_pred[:, :-1] - np.expand_dims(y_test, axis=1)
        closest_q_idx = np.argmin(np.abs(diffs), axis=1)
        closest_q = np.array([taus[closest_q_idx[i]] for i in range(y_test.shape[0])])

    for i in range(n_quantiles):
        a = taus[i-1] if i > 0 else 0.0
        b = taus[i]
        
        # Uncensored data points
        if b < 1.0:
            smaller_b = y_test[cen_indicator_test == 0] <= y_test_pred[cen_indicator_test == 0, i] 
            smaller_b_cens = y_test[cen_indicator_test == 1] <= y_test_pred[cen_indicator_test == 1, i]
        else:
            smaller_b =  y_test[cen_indicator_test == 0] < 1e9
            smaller_b_cens = y_test[cen_indicator_test == 1] < 1e9
        
        if a > 0.0:
            larger_a = y_test[cen_indicator_test == 0] >= y_test_pred[cen_indicator_test == 0, i-1]
            larger_a_cens = y_test[cen_indicator_test == 1] >= y_test_pred[cen_indicator_test == 1, i-1]
            smaller_a_cens = y_test[cen_indicator_test == 1] <= y_test_pred[cen_indicator_test == 1, i-1] 
        else:
            larger_a = y_test[cen_indicator_test == 0] >= -1e9 
            larger_a_cens = y_test[cen_indicator_test == 1] >= -1e9 
            smaller_a_cens = y_test[cen_indicator_test == 1] <= -1e9
        
        fallwithin = smaller_b * larger_a
        fallwithin_cens = smaller_b_cens * larger_a_cens
        cens_part1 = fallwithin_cens * (b - closest_q[cen_indicator_test == 1]) / (1 - closest_q[cen_indicator_test == 1])
        cens_part2 = smaller_a_cens * (b - a) / (1 - closest_q[cen_indicator_test == 1])

        total_points = fallwithin.sum() + cens_part1.sum() + cens_part2.sum()
        prop_captured = total_points / y_test.shape[0]

        dcal_data_cens.append([b - a, prop_captured])
    dcal_data_cens = np.array(dcal_data_cens)

    return np.cumsum(dcal_data_cens[:, 1])


def get_ground_truth(dataset_str, x_test):
    """
    Calculate and return the true mean and variance for the train and test datasets
    based on the specified dataset type.
    Parameters:
        dataset_str (str): The name of the dataset type.
        x_test (ndarray): Input data for the testing set.
    Returns:
        true_mean_test (ndarray): True mean for the testing set.
        true_sd_test (ndarray): True stand deviation for the testing set.
    """

    if dataset_str == 'Gaussian_linear':
        # True mean and variance for a linear Gaussian dataset
        true_mean_test = 2 * x_test + 10
        true_sd_test = x_test + 1

    elif dataset_str == 'Gaussian_nonlinear':
        # True mean and variance for a nonlinear Gaussian dataset
        true_mean_test = x_test * np.sin(2 * x_test) + 10
        true_sd_test = (x_test + 1) / 2

    elif dataset_str == 'Exponential':
        # True mean and variance for an Exponential dataset
        true_mean_test = 2 * x_test + 4
        true_sd_test = 2 * x_test + 4

    elif dataset_str == 'LogNorm':
        # True mean and variance for a LogNormal dataset
        true_mean_test = np.exp((x_test - 1) ** 2 + 1 / 2)
        true_sd_test = np.sqrt((np.exp(1) - 1) * np.exp(2 * (x_test - 1) ** 2 + 1))

    elif dataset_str == 'Weibull':
        # True mean and variance for a Weibull dataset
        true_mean_test = (x_test * np.sin(2 * (x_test - 1)) * 4 + 10) * sps.gamma(1 + 1 / 5)
        true_sd_test = np.sqrt((x_test * np.sin(2 * (x_test - 1)) * 4 + 10) ** 2 * (sps.gamma(1 + 2 / 5) - sps.gamma(1 + 1 / 5) ** 2))

    elif dataset_str == 'Gaussian_uniform':
        # True mean and variance for a Gaussian dataset with uniform noise
        true_mean_test = 2 * x_test * np.cos(2 * x_test) + 13
        true_sd_test = x_test ** 2 + 1 / 2

    elif dataset_str in ['Norm_med', 'Norm_heavy', 'Norm_light', 'Norm_same']:
        # True mean and variance for different types of Gaussian datasets
        true_mean_test = (x_test[:, 0] * 3 + x_test[:, 1] ** 2 - x_test[:, 2] ** 2 + np.sin(x_test[:, 3] * x_test[:, 2]) + 6).reshape([1, -1]).T
        true_sd_test = (x_test[:, 0] * 0 + 1).reshape([1, -1]).T

    elif dataset_str in ['LogNorm_med', 'LogNorm_heavy', 'LogNorm_light', 'LogNorm_same']:
        # True mean and variance for different types of LogNormal datasets
        betas = np.array([[0.8, 0.6, 0.4, 0.5, -0.3, 0.2, 0.0, -0.7]]).T
        true_mean_test = np.exp((np.matmul(x_test, betas)).reshape([1, -1]).T - np.log(10) + 1 / 2)
        true_sd_test = np.sqrt((np.exp(0.5 ** 2) - 1) * np.exp(2 * (np.matmul(x_test, betas)).reshape([1, -1]).T - 2 * np.log(10) + 1))

    return true_mean_test, true_sd_test


def aggregate(model_str, dataset_str, results):
    aggregated_results = {}
    for key in results[0].keys():
        values = [m[key] for m in results]  
        mean = np.mean(values)             
        std = np.std(values)               
        aggregated_results[key] = f"{mean:.3f} ± {std:.3f}"  
    final_results = {
        "aggregated_results": aggregated_results,
        "original_results": results
    }
    os.makedirs(f'res/{model_str}/', exist_ok=True)  # Ensure the directory exists
    with open(f'res/{model_str}/{dataset_str}.json', 'w') as file:
        json.dump(final_results, file, indent=4, ensure_ascii=False)
    return None

def get_median_durations(model_str, surv, y_median=None, y_train=None, num_durations=None):
    """
    Calculate the duration corresponding to a survival probability of 0.5 for each sample.

    Args:
        surv (pd.DataFrame): DataFrame where rows are durations and columns are samples.

    Returns:
        np.ndarray: An array containing the median survival time for each sample.
                     If no duration corresponds to a survival probability of 0.5 or less, the value is None.
    """
    if model_str == "DeepSurv":
        # Extract the durations (index) as a NumPy array
        durations = surv.index.to_numpy()

        # List to store median survival times for each sample
        median_durations = []

        # Iterate over each sample (column in the DataFrame)
        for col in surv.columns:
            prob = 1 - surv[col].to_numpy()  # Get the survival probabilities for the sample
            idx = (prob >= 0.5).argmax()  # Find the first index where probability <= 0.5
            if prob[idx] >= 0.5:  # Ensure the condition is actually met
                median_durations.append(durations[idx])
            else:
                median_durations.append(y_median)  # No duration reached 0.5
        # Convert the results into a NumPy array
        return np.array(median_durations)
    if model_str == 'DeepHit':
        bins = np.linspace(y_train.min(), y_train.max(), num_durations + 1)  
        durations = (bins[:-1] + bins[1:]) / 2  
        # durations = surv.index.to_numpy()

        # List to store median survival times for each sample
        median_durations = []

        # Iterate over each sample (column in the DataFrame)
        for col in surv.columns:
            prob = 1 - surv[col].to_numpy()  # Get the survival probabilities for the sample
            idx = (prob >= 0.5).argmax()  # Find the first index where probability <= 0.5
            if prob[idx] >= 0.5:  # Ensure the condition is actually met
                median_durations.append(durations[idx])
            else:
                median_durations.append(np.median(y_train))  # No duration reached 0.5
        # Convert the results into a NumPy array
        return np.array(median_durations)


def get_median_survival_time(sf):
    x_vals = sf.x
    y_vals = sf.y
    idx = np.where(y_vals <= 0.5)[0]

    if len(idx) == 0:
        return x_vals[-1]  
    else:
        first_idx = idx[0]
        return x_vals[first_idx]


def get_median_dsm_survival_time(sf):
    x_vals = sf.x
    y_vals = sf.y
    idx = np.where(y_vals >= 0.5)[0]

    if len(idx) == 0:
        return x_vals[-1]  
    else:
        first_idx = idx[0]
        return x_vals[first_idx]
