import os
import torch
import h5py
import numpy as np
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from pdb import set_trace as bb
from sklearn import preprocessing
from collections import defaultdict
from sklearn.model_selection import train_test_split


# this file contains synthetic dataset classes
def get_dataset(dataset_str):
	if dataset_str == 'Gaussian_linear':
		mydataset = GaussianLinear()
	elif dataset_str == 'Gaussian_nonlinear':
		mydataset = GaussianNonLinear()
	elif dataset_str == 'Exponential':
		mydataset = Exponential()
	elif dataset_str == 'LogNorm':
		mydataset = LogNorm()
	elif dataset_str == 'Weibull':
		mydataset = Weibull()
	elif dataset_str == 'Gaussian_uniform':
		mydataset = GaussianUniform()
	elif dataset_str == 'Norm_med':
		mydataset = Norm_med()
	elif dataset_str == 'Norm_heavy':
		mydataset = Norm_heavy()
	elif dataset_str == 'Norm_light':
		mydataset = Norm_light()
	elif dataset_str == 'Norm_same':
		mydataset = Norm_same()
	elif dataset_str == 'LogNorm_med':
		mydataset = LogNorm_med()
	elif dataset_str == 'LogNorm_heavy':
		mydataset = LogNorm_heavy()
	elif dataset_str == 'LogNorm_light':
		mydataset = LogNorm_light()
	elif dataset_str == 'LogNorm_same':
		mydataset = LogNorm_same()
	elif dataset_str == 'METABRIC':
		mydataset = METABRIC()
	elif dataset_str == 'WHAS':
		mydataset = WHAS()
	elif dataset_str == 'SUPPORT':
		mydataset = SUPPORT()
	elif dataset_str == 'GBSG':
		mydataset = GBSG()
	elif dataset_str == 'TMBImmuno':
		mydataset = TMBImmuno()
	elif dataset_str == 'BreastMSK':
		mydataset = BreastMSK()
	elif dataset_str == 'LGGGBM':
		mydataset = LGGGBM()
	else:
		raise Exception(dataset_str + 'dataset not defined')
	return mydataset

def generate_data_synthtarget_synthcen(n_data, mydataset, x_range=[0,2], is_censor=True):
	# is_censor=False allows to only draw data from observed dist
	# this fn is used if we're generating everything, data and censoring
	"""
	Parameters
	----------
	n_data: the number of data
	mydataset: the name of dataset
	x_range=[0,2]: the range of x, x ~ U(0, 2)
	taus: the given quantile, e.g. 0.1, ..., 0.9
	is_censor: whether to consider censored data
	x: the input of dataset
	y: the label of dataset
	y_taus: the specific y_hat under the given taus
	----------
	"""

	# sample x
	x = np.random.uniform(x_range[0], x_range[1], size=(n_data, mydataset.input_dim)) 

	# compute target
	target = mydataset.get_observe_times(x).flatten()

	# compute censor
	if is_censor:
		cen = mydataset.get_censor_times(x).flatten()
	else:
		cen = target # otherwise make censored times larger than observed
	
	# y = min(target, cen)
	y, cen_indicator, obs_indicator = mydataset.process_censor_observe(target, cen)
	
	# obtain the targeted y_taus with the given taus
	# y_taus = [np.percentile(y, t * 100) for t in taus]
	# y_taus = [np.quantile(target[obs_indicator.flatten().astype(bool)], t) for t in taus]
	# y_taus = [np.quantile(target, t) for t in taus]

	return x, target.reshape(-1, 1), cen.reshape(-1, 1), y.reshape(-1, 1), cen_indicator.reshape(-1, 1), obs_indicator.reshape(-1, 1)

def generate_data_realtarget_realcen(mydataset, test_propotion, rand_in):
	x_train, x_test, y_train, y_test, cen_indicator_train, cen_indicator_test = mydataset.get_data(test_propotion, rand_in)

	data_train = (x_train, y_train.reshape(-1, 1), y_train.reshape(-1, 1), y_train.reshape(-1, 1), cen_indicator_train.reshape(-1, 1), np.abs(cen_indicator_train-1).reshape(-1, 1))
	data_test = (x_test, y_test.reshape(-1, 1), y_test.reshape(-1, 1), y_test.reshape(-1, 1), cen_indicator_test.reshape(-1, 1), np.abs(cen_indicator_test-1).reshape(-1, 1))
	
	return data_train, data_test

# parent class for all the datasets
class DataSet:
	"""
	Parameters
	----------
	target: the target variable, t_i ~ p_t(t|x_i)
	censor: the censoring variable, c_i ~ p_c(c|x_i)
	y: the label of dataset, y_i = min (t_i, c_i)
	----------
	"""
	# Get label of dataset
	def process_censor_observe(self, target, censor):
		y = np.minimum(censor, target)
		cen_indicator = np.array([censor < target]) * 1 # 1 if censored else 0
		obs_indicator = np.array([censor >= target]) * 1 # 1 if observed else 0
		return y, cen_indicator, obs_indicator
	
# parent class for synthetic datasets
class SyntheticDataSet(DataSet):
	"""
	Parameters
	----------
	x: the input variable of dataset, x \in R^D, D is the dimension of the input
	q: the given quantile, e.g. 0.1, ..., 0.9
	----------
	"""
	def __init__(self):
		self.synth_target = True # whether timetevent is synthetically generated
		self.synth_censor = True # whether censoring is synthetically generated
	def get_observe_times(self, x):
		pass
	def get_censor_times(self, x):
		pass
	def get_quantile_truth(self, x, q):
		pass
	def get_mean_truth(self, x):
		pass
	
# parent class for synthetic 1D datasets
class SyntheticDataSet1D(SyntheticDataSet):
	def __init__(self):
		super().__init__() 
		self.input_dim=1
	    	
# parent class for Gaussian datasets 
# Y ~ N(self.param1, self.param2)
class Gaussian(SyntheticDataSet1D):
	def get_observe_times(self, x):
		return np.random.normal(loc=self.param1_target(x), scale=self.param2_target(x))
	def get_censor_times(self, x):
		return np.random.normal(loc=self.param1_cen(x), scale=self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_target(x), self.param2_target(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.norm(self.param1_target(x), self.param2_target(x)).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_cen(x), self.param2_cen(x)).ppf(q)

# Gaussian Linear dataset
class GaussianLinear(Gaussian):
	def param1_target(self, x):
		return x * 2 + 10
	def param2_target(self, x):
		return (x + 1) / 1
	def param1_cen(self, x):
		return x * 4 + 10
	def param2_cen(self, x):
		return (x * 4 + 2) / 5

# Gaussian Non Linear dataset
class GaussianNonLinear(GaussianLinear):
	def param1_target(self, x):
		return x * np.sin(x * 2) + 10
	def param2_target(self, x):
		return (x + 1) / 2
	def param1_cen(self, x):
		return x * 2 + 10
	def param2_cen(self, x):
		return 2

# Exponential dataset
class Exponential(SyntheticDataSet1D):
	def param1_target(self,x):
		return 2*x+4
	def param1_cen(self,x):
		# return x*3+10
		# return 20-x*3
		return 15-x*3
	def get_observe_times(self, x):
		return np.random.exponential(scale=self.param1_target(x))
	def get_censor_times(self, x):
		return np.random.exponential(scale=self.param1_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.expon(scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.expon(scale=self.param1_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.expon(scale=self.param1_target(x)).mean()
	
class LogNorm(SyntheticDataSet1D):
	def get_observe_times(self, x):
		# return np.random.exponential(scale=self.param1_target(x))
		return np.random.lognormal(mean=self.param1_target(x), sigma=self.param2_target(x))
	def get_censor_times(self, x):
		# return np.random.exponential(scale=self.param1_cen(x))
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		# return scipy.stats.expon(scale=self.param1_target(x)).ppf(q)
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).ppf(q)
		# return scipy.stats.lognorm(s=self.param2_target(x),scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		# return scipy.stats.expon(scale=self.param1_cen(x)).ppf(q)
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).mean()
	def param1_target(self,x):
		return (x-1)**2
	def param2_target(self,x):
		# return x
		# return x*0 + 0.5
		return x*0 + 1
	def param1_cen(self,x):
		return x*0 
	def param2_cen(self,x):
		return x*0 + 10
	
class Weibull(SyntheticDataSet1D):
	def __init__(self):
		super().__init__() 
		self.weibull_shape = 5 # =1 is exponential, could get fancy and make this depend on x
	def param1_target(self,x):
		return x*np.sin(2*(x-1))*4+10
		# return self.c
	def param1_cen(self,x):
		return 20-x*3
		# return self.c
	def get_observe_times(self, x):
		return self.param1_target(x)*np.random.weibull(a=self.weibull_shape, size=x.shape)
	def get_censor_times(self, x):
		return self.param1_cen(x)*np.random.weibull(a=self.weibull_shape, size=x.shape)
	def get_quantile_truth(self, x, q):
		return scipy.stats.weibull_min(c=self.weibull_shape, scale=self.param1_target(x)).ppf(q)
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.weibull_min(c=self.weibull_shape, scale=self.param1_cen(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.weibull_min(c=self.weibull_shape, scale=self.param1_target(x)).mean()

class GaussianUniform(SyntheticDataSet1D):
	# target is Gaussian, censored is Uniform
	def get_observe_times(self, x):
		# return np.maximum(np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x)),0.1)
		# return np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x))
		return np.clip(np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x)),0.1,10000)
	def get_censor_times(self, x):
		# return np.maximum(np.random.normal(loc=self.param1_cen(x), scale = self.param2_cen(x)),0.1)
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	
class GaussianUniform(GaussianUniform):
	def param1_target(self,x):
		return 2*x*np.cos(2*x)+13
	def param2_target(self,x):
		return (x**2+1/2)
	def param1_cen(self,x):
		return x*0
	def param2_cen(self,x):
		return x*0+18 # this is width
	
# LogNorm_med dataset
class LogNorm_med(SyntheticDataSet):
	def __init__(self):
		super().__init__() 
		self.input_dim=8
		self.betas = np.array([[0.8, 0.6, 0.4, 0.5, -0.3, 0.2, 0.0, -0.7]]).T
		# self.betas = np.array([[0.1, 0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.3]]).T
		x = np.random.uniform(0,2,size=(10000,self.input_dim))
		self.y_max_cens = np.quantile(self.get_observe_times(x),0.95)
	def get_observe_times(self, x):
		return np.random.lognormal(mean=self.param1_target(x), sigma=self.param2_target(x))/10
		# the mean and standard deviation are not the values for the distribution itself, but of the underlying normal distribution it is derived from.
	def get_censor_times(self, x):
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		# https://stackoverflow.com/questions/8870982/how-do-i-get-a-lognormal-distribution-in-python-with-mu-and-sigma
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.lognorm(s=self.param2_target(x),scale=np.exp(self.param1_target(x))).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def param1_target(self,x):
		return np.matmul(x, self.betas).squeeze()
	def param2_target(self,x): 
		# return x[:,0]*0 + 0.5 
		return x[:,0]*0 + 1.
	def param1_cen(self,x):
		return x[:,0]*0 
	def param2_cen(self,x):
		return x[:,0]*0 + 1.

# LogNorm_light dataset
class LogNorm_light(LogNorm_med):
	def param2_cen(self,x):
		return x[:,0]*0 + 3.5

# LogNorm_heavy dataset
class LogNorm_heavy(LogNorm_med):
	def param2_cen(self,x):
		return x[:,0]*0 + 0.4

# LogNorm_same dataset
class LogNorm_same(LogNorm_med):
	def get_censor_times(self, x):
		return self.get_observe_times(x)

class Norm_med(SyntheticDataSet):
	def __init__(self):
		super().__init__() 
		self.input_dim=4
		self.offset = 0
		x = np.random.uniform(0,2,size=(10000,self.input_dim))
		self.offset = round(-min(self.get_observe_times(x))+1, 3)	
		print('self.offset',self.offset)
	def get_observe_times(self, x):
		return np.random.normal(loc=self.param1_target(x), scale = self.param2_target(x))
	def get_censor_times(self, x):
		return np.random.uniform(self.param1_cen(x), self.param1_cen(x)+self.param2_cen(x))
	def get_quantile_truth(self, x, q):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).ppf(q)
	def get_mean_truth(self, x):
		return scipy.stats.norm(self.param1_target(x),self.param2_target(x)).mean()
	def get_censored_quantile_truth(self, x, q):
		return scipy.stats.uniform(loc=self.param1_cen(x),scale=self.param2_cen(x)).ppf(q)
	def param1_target(self,x):
		return x[:,0]*3+x[:,1]**2-x[:,2]**2+np.sin(x[:,3]*x[:,2]) + 6
	def param2_target(self,x):
		# return x[:,0]*0+0.5
		return x[:,0]*0 + 1.
	def param1_cen(self,x):
		return x[:,0]*0 + 0
	def param2_cen(self,x):
		return x[:,0]*0 + 20

class Norm_heavy(Norm_med):
	# we lower the censoring point so that most of the target dist is undefined
	# this is heavy censoring
	def param2_cen(self,x):
		return x[:,0]*0 + 12
	
class Norm_light(Norm_med):
	# light censoring
	def param2_cen(self,x):
		return x[:,0]*0 + 40

class Norm_same(Norm_med):
	# same as for target dist
	def get_censor_times(self, x):
		return self.get_observe_times(x)


class RealTargetRealCensor(DataSet):
	def __init__(self):
		self.synth_target = False
		self.synth_censor = False

	def vis_data(self):
		nshow=1000
		fig, ax = plt.subplots(self.input_dim,1)
		for i in range(self.input_dim):
			# ax[i].scatter(self.df.data[:nshow,i], self.df.target[:nshow,0],s=6, alpha=0.5)
			ax[i].scatter(self.data[:nshow,i][self.target[:nshow,1] == 0], self.target[:nshow,0][self.target[:nshow,1] == 0],color='g',marker='+',s=20,label='observed')
			ax[i].scatter(self.data[:nshow,i][self.target[:nshow,1] == 1], self.target[:nshow,0][self.target[:nshow,1] == 1],color='g',marker='^',s=10,label='censored')
		fig.show()

	def get_data(self, test_propotion, rand_in):

		# get a random test/train split
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=test_propotion, random_state=rand_in)

		# !!! subselect for development speed
		n_subselect = 500000
		self.x_train, self.x_test, self.y_train, self.y_test = self.x_train[:n_subselect], self.x_test[:n_subselect], self.y_train[:n_subselect], self.y_test[:n_subselect]

		# note that y_train contains both target and censored columns, so we split here
		self.cen_train_indicator = self.y_train[:,1].reshape(1,-1)
		self.cen_test_indicator = self.y_test[:,1].reshape(1,-1)
		self.y_train = self.y_train[:,0].flatten()
		self.y_test = self.y_test[:,0].flatten()

		return self.x_train, self.x_test, self.y_train, self.y_test, self.cen_train_indicator, self.cen_test_indicator

class METABRIC(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		# self.df = pd.read_csv(os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5'))
		dataset_file = os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		if False: # optionally visualise
			self.vis_data()
		return

class WHAS(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		# self.df = pd.read_csv(os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5'))
		dataset_file = os.path.join(path_data,'whas_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()

		self.target[:,0]-=self.target[:,0].min()-1e-1 # adjust so no negative times

		if False: # optionally visualise
			self.vis_data()
		return

class GBSG(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 
		# note that the way I set up this exp is closer to Deep Extended Hazard Models for Survival Analysis
		# and not comparable w deepsurv paper since they use test/train split as per diff studies

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		# self.df = pd.read_csv(os.path.join(path_data,'metabric_IHC4_clinical_train_test.h5'))
		dataset_file = os.path.join(path_data,'gbsg_cancer_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 5
		self.data = np.clip(self.data,-x_lim,x_lim)
		self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return

class SUPPORT(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 
		# note that the way I set up this exp is closer to Deep Extended Hazard Models for Survival Analysis
		# and not comparable w deepsurv paper since they use test/train split as per diff studies

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		dataset_file = os.path.join(path_data,'support_train_test.h5')

		# datasets and opening code borrowed from https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/utils.py
		datasets = defaultdict(dict)
		with h5py.File(dataset_file, 'r') as fp:
			for ds in fp:
				for array in fp[ds]:
					datasets[ds][array] = fp[ds][array][:]

		# for my exps, I merge test and train splits together and resplit later
		self.target = np.concatenate([datasets['test']['t'],datasets['train']['t']])
		self.data = np.concatenate([datasets['test']['x'],datasets['train']['x']],axis=0)
		self.event = np.concatenate([datasets['test']['e'],datasets['train']['e']]) # in this dataset 1=observed, 0=censored

		# we concat target with event
		self.target = np.stack([self.target,self.event]).T 

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)
		self.input_dim=self.data.shape[1]

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		if False: # optionally visualise
			self.vis_data()
		return

class TMBImmuno(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		# http://www.cbioportal.org/study/clinicalData?id=tmb_mskcc_2018

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		self.df=pd.read_table(os.path.join(path_data,'tmb_immuno_mskcc.tsv'),sep='\t')

		event_arr = np.array(self.df['Overall Survival Status'])
		self.df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
		self.df['time'] = np.array(self.df['Overall Survival (Months)']).astype(np.float64)

		self.df['age_new'] = np.array(self.df['Age at Which Sequencing was Reported (Days)'])
		sex_arr = np.array(self.df['Sex'])
		self.df['sex_new'] = np.array([1 if sex_arr[i] == 'Female' else 0 for i in range(sex_arr.shape[0])])

		# remove nans
		self.df.dropna(subset = ['event', 'time', 'age_new','sex_new','TMB (nonsynonymous)'],how='any',inplace=True)
	
		# use self.target instead of self.df.target to avoid pandas warning
		self.target = pd.concat([self.df.pop(x) for x in ['time','event']], axis=1)
		self.data = pd.concat([self.df.pop(x) for x in ['age_new','sex_new','TMB (nonsynonymous)']], axis=1)


		self.target = np.array(self.target)
		self.data = np.array(self.data)

		self.input_dim=self.data.shape[1]

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 5
		self.data = np.clip(self.data,-x_lim,x_lim)
		self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return

		

class BreastMSK(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 
		# https://www.cbioportal.org/study/summary?id=breast_msk_2018

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		self.df=pd.read_table(os.path.join(path_data, 'breast_msk_2018_clinical_data.tsv'),sep='\t')

		event_arr = np.array(self.df['Overall Survival Status'])
		self.df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
		self.df['time'] = self.df['Overall Survival (Months)']

		tmp_arr = np.array(self.df['ER Status of the Primary'])
		self.df['ER_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])
		tmp_arr = np.array(self.df['Overall Patient HER2 Status'])
		self.df['HER2_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])
		tmp_arr = np.array(self.df['Overall Patient HR Status'])
		self.df['HR_new'] = np.array([1 if tmp_arr[i] == 'Positive' else 0 for i in range(tmp_arr.shape[0])])

		# remove nans
		self.df.dropna(subset = ['event', 'time', 'ER_new', 'HER2_new', 'HR_new', 'Mutation Count', 'TMB (nonsynonymous)'],how='any',inplace=True)
	
		# use self.target instead of self.df.target to avoid pandas warning
		self.target = pd.concat([self.df.pop(x) for x in ['time', 'event']], axis=1)
		self.data = pd.concat([self.df.pop(x) for x in ['ER_new', 'HER2_new', 'HR_new', 'Mutation Count', 'TMB (nonsynonymous)']], axis=1)

		self.target = np.array(self.target)
		self.data = np.array(self.data)

		self.input_dim=self.data.shape[1]

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 5
		self.data = np.clip(self.data,-x_lim,x_lim)
		self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return

class LGGGBM(RealTargetRealCensor):
	def __init__(self):
		super().__init__() 

		# https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub

		path_cd = os.getcwd()
		path_project = os.path.dirname(path_cd) # up one level
		path_data = os.path.join(path_project,'Experiment/datasets')
		self.df=pd.read_table(os.path.join(path_data,'lgggbm_tcga_pub_clinical_data.tsv'),sep='\t')

		# remove nans
		self.df.dropna(subset = ['Overall Survival Status', 'Overall Survival (Months)', 'Diagnosis Age','Sex','Absolute Purity','Mutation Count','TMB (nonsynonymous)'],how='any',inplace=True)

		event_arr = np.array(self.df['Overall Survival Status'])
		self.df['event'] = np.array([int(event_arr[i][0]) for i in range(event_arr.shape[0])])
		self.df['time'] = np.array(self.df['Overall Survival (Months)']).astype(np.float64)

		tmp_arr = np.array(self.df['Sex'])
		self.df['sex_new'] = np.array([1 if tmp_arr[i] == 'Female' else 0 for i in range(tmp_arr.shape[0])])
	
		# use self.target instead of self.df.target to avoid pandas warning
		self.target = pd.concat([self.df.pop(x) for x in ['time','event']], axis=1)
		self.data = pd.concat([self.df.pop(x) for x in ['Diagnosis Age','sex_new','Absolute Purity','Mutation Count','TMB (nonsynonymous)']], axis=1)

		self.target = np.array(self.target)
		self.data = np.array(self.data)

		self.input_dim=self.data.shape[1]

		# in dataset 1=observed, 0=censored, so invert this
		self.target[:,1] = np.abs(self.target[:,1]-1)

		# mean zero, unit variance, for features and target
		x_scaler = preprocessing.StandardScaler().fit(self.data)
		self.data = x_scaler.transform(self.data)
		y_scaler = preprocessing.StandardScaler().fit(self.target[:,0].reshape(-1, 1))
		self.target[:,0] = y_scaler.transform(self.target[:,0].reshape(-1, 1)).flatten()
		self.target[:,0]-=self.target[:,0].min()-1e-1

		# clip outliers
		x_lim = 3
		self.data = np.clip(self.data,-x_lim,x_lim)
		# self.target = np.clip(self.target,-x_lim,x_lim)

		if False: # optionally visualise
			self.vis_data()
		return
