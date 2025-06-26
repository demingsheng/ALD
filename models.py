import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bb
from scipy.stats import lognorm


class MLP_ALD(nn.Module):
	def __init__(self, input_dim, n_hidden, output_dim, is_dropout=False, dropout_rate=None):
		super(MLP_ALD, self).__init__()
		self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
		self.layer2_theta = nn.Linear(n_hidden, n_hidden, bias=True)
		self.layer2_sigma = nn.Linear(n_hidden, n_hidden, bias=True)
		self.layer2_kappa = nn.Linear(n_hidden, n_hidden, bias=True)
		self.layer3_theta = nn.Linear(n_hidden, output_dim, bias=True)
		self.layer3_sigma = nn.Linear(n_hidden, output_dim, bias=True)
		self.layer3_kappa = nn.Linear(n_hidden, output_dim, bias=True)
		self.projection = nn.Linear(input_dim, n_hidden, bias=False)
		self.is_dropout = is_dropout
		self.dropout_rate = dropout_rate 
		if self.is_dropout:
			self.dropout = nn.Dropout(dropout_rate)
	
	def forward(self, x):
		residual = self.projection(x)
		x = torch.relu(self.layer1(x))
		x = x + residual
		if self.is_dropout:
			x = self.dropout(x)

		theta = torch.relu(self.layer2_theta(x))
		if self.is_dropout:
			theta = self.dropout(theta)
		# theta = torch.exp(self.layer3_theta(theta)) # Force ALD's mode to be positive
		theta = self.layer3_theta(theta)

		sigma = torch.relu(self.layer2_sigma(x))
		if self.is_dropout:
			sigma = self.dropout(sigma)
		sigma = torch.exp(self.layer3_sigma(sigma)) 

		kappa = torch.relu(self.layer2_kappa(x))
		if self.is_dropout:
			kappa = self.dropout(kappa)
		kappa = torch.exp(self.layer3_kappa(kappa))

		return theta, sigma, kappa

	def ald_cdf(self, bx, by, gamma=None):
		theta, sigma, kappa = self.forward(bx)
		if gamma is not None:
			theta, sigma, kappa = theta*gamma[:,0:1], sigma*gamma[:,1:2], kappa*gamma[:,2:3]

		ald_cdf_1 = 1 - 1 / (1 + kappa**2) * torch.exp(-torch.sqrt(torch.tensor(2.0)) * kappa * (by - theta) / sigma)
		ald_cdf_2 = kappa**2 / (1 + kappa**2) * torch.exp(-torch.sqrt(torch.tensor(2.0)) * (theta - by) / (sigma * kappa))
		ald_cdf = torch.where(by > theta, ald_cdf_1, ald_cdf_2)
		
		return ald_cdf
	

class MLP_Base(nn.Module):
	def __init__(self, input_dim, n_hidden, output_dim, is_dropout=False, dropout_rate=None, is_positive=False):
		super(MLP_Base, self).__init__()
		self.layer1 = nn.Linear(input_dim, n_hidden, bias=True)
		self.layer2 = nn.Linear(n_hidden, n_hidden, bias=True)
		self.layer3 = nn.Linear(n_hidden, output_dim, bias=True) 
		self.is_dropout = is_dropout
		self.is_positive = is_positive
		if self.is_dropout:
			self.dropout = nn.Dropout(dropout_rate)


	def forward(self,x):
		x = torch.relu(self.layer1(x))
		if self.is_dropout:
			x = self.dropout(x)

		x = torch.relu(self.layer2(x))
		if self.is_dropout:
			x = self.dropout(x)

		if self.is_positive:
			x = torch.exp(self.layer3(x))
		else:
			x = self.layer3(x)

		return x
		
	
def safe_log(x, min_value=1e-8):
    """Numerically stable logarithm."""
    return torch.log(torch.clamp(x, min=min_value))


def loss_ald(y, theta, sigma, kappa, cen_indicator, is_use_censor_loss=True):
	"""
    Computes the loss for Asymmetric Laplace Distribution under censoring.
    Parameters:
        y (Tensor): Target variable.
        theta, sigma, kappa (Tensor): Parameters of the ALD.
        cen_indicator (Tensor): 0 = observed, 1 = censored.
        is_use_censor_loss (Bool): Whether to include censoring loss.
    Returns:
        Tensor: Loss value.
    """
	alpha = (y >= theta) * (y - theta)
	beta = (y < theta) * (theta - y)
	loss_obs = torch.sum((cen_indicator==0) * (torch.log(sigma) - torch.log((kappa)/(1+kappa**2)) + (torch.sqrt(torch.tensor([2]))/sigma)*(alpha*kappa + beta/kappa))).mean()
    
	if not is_use_censor_loss:
		loss = loss_obs
	else:
		loss_cen_1 = safe_log(1 + kappa**2) + torch.sqrt(torch.tensor([2])) * kappa * (y - theta) / sigma
		value_to_exp = - torch.sqrt(torch.tensor([2])) * (theta - y) / (sigma*kappa)
		safe_exp_value = torch.clamp(value_to_exp, max=80)
		loss_cen_2 = - safe_log(1 - kappa**2 * torch.exp(safe_exp_value)/(kappa**2+1))
		loss_cen = torch.sum((cen_indicator == 1) * torch.where(y > theta, loss_cen_1, loss_cen_2)).mean()
		loss = loss_obs + loss_cen

	return loss


def loss_cqr(y, y_pred, y_max, cen_indicator, is_use_censor_loss=True):
    """
    Censored Quantile Regression Loss.
    Parameters:
		y (Tensor): Ground-truth times, shape (B,) or (B,1)
        y_pred (Tensor): Predicted quantiles, shape (B, Q)
		y_max (float): pseudo value
        cen_indicator (Tensor): 0 if observed, 1 if censored, shape (B,)
        q (array-like or Tensor): Quantile levels, shape (Q,)
        is_use_censor_loss (bool): Whether to include censored loss
    Returns:
        Scalar tensor: combined loss
    """
    B, Q = y_pred.shape

    taus = np.linspace(1/Q,1,Q)

    # Build a (B, Q) tensor of quantile levels
    tau_block = torch.tensor(taus, dtype=y_pred.dtype, device=y_pred.device) \
                        .unsqueeze(0) \
                        .repeat(B, 1)  # shape (B, Q)

    # 1. Observed loss
    y = y.view(-1, 1)  # ensure shape (B, 1)
    indicators = (y_pred < y).float()
    loss_obs = ( (cen_indicator == 0).float().unsqueeze(1)
                 * (y_pred - y)
                 * ((1 - tau_block) - indicators) )
    loss_obs = loss_obs.sum(dim=1).mean()

    if not is_use_censor_loss:
        return loss_obs

    # 2. Censored loss: estimate closest quantile (ignore last quantile)
    with torch.no_grad():
        abs_diffs = torch.abs(y - y_pred[:, :-1])   # (B, Q-1)
        min_abs   = abs_diffs.min(dim=1, keepdim=True).values
        mask      = (abs_diffs == min_abs)          # (B, Q-1)
        est_q     = (tau_block[:, :-1] * mask).max(dim=1).values  # (B,)

    # 3. Portnoy weights
    eq_q      = est_q.unsqueeze(1)                        # (B,1)
    one_minus = torch.clamp(1 - eq_q, min=1e-6)            # avoid div0
    weights   = (tau_block[:, :-1] < eq_q).float() + \
                (tau_block[:, :-1] >= eq_q) * \
                ((tau_block[:, :-1] - eq_q) / one_minus)

    # 4. Censored loss computation
    indic_cens = (y_pred[:, :-1] < y).float()
    loss_cens  = weights * (y_pred[:, :-1] - y) * ((1 - tau_block[:, :-1]) - indic_cens) \
               + (1 - weights) * (y_pred[:, :-1] - y_max) * ((1 - tau_block[:, :-1]) - (y_pred[:, :-1] < y_max).float())

    loss_cens = (cen_indicator > 0).float().unsqueeze(1) * loss_cens
    loss_cens = loss_cens.sum(dim=1).mean()

    return loss_obs + loss_cens


def loss_lognorm(y_true, y_pred, cen_indicator):
	# y_pred is now shape batch_size, 2
	# representing mean and stddev of log normal dist
	# (but stddev needs transforming w softplus)
	# logT = N(mean, stddev^2)
	# for observed data points, want to minimise -logpdf
	# for censored, want to minimise -logcdf
	# this helped: https://github.com/rajesh-lab/X-CAL/blob/master/models/loss/mle.py

	mean = y_pred[:,0]
	soft_fn = nn.Softplus()
	stddev = soft_fn(y_pred[:,1])

	pred_dist = torch.distributions.LogNormal(mean,stddev)

	logpdf = torch.diagonal(pred_dist.log_prob(y_true))
	cdf = torch.diagonal(pred_dist.cdf(y_true))
	logsurv = torch.log(1.0-cdf+1e-4)

	loglike = torch.mean((cen_indicator<1)*logpdf + (cen_indicator>0)*logsurv)
	loss = -loglike

	return loss
