import numpy as np


# this file contains hyperparams for benchmarking experiments, after tuning
def get_hyperparams(dataset_str, loss_str):
	"""
	Parameters
	----------
	dataset_str: the name of datasets, e.g. Norm linear
	loss_str: the name of loss functions, e.g. cqrnn
	----------
	"""
	# create a dict of hyperparams
	hyp = {}
	
	hyp = 	{'batch_size': 128,
			 'n_epochs': 100,
			 'learning_rate': 0.001,
			 'weight_decay': 1e-4,
			 'model_str': 'mlp', # what architecture to use
			 'is_dropout': False, # whether to use dropout
			 'is_batch': False, # whether to use batchnorm
			 'n_hidden': 8, # hidden nodes in NN
			 'n_quantiles': 100, # how many quantiles to predict
			 'test_propotion': 0.2, # proportion of dataset to use for test (real data)
			 'y_max': 99., # value to use for large pseudo datapoint in cqrnn loss
			 'n_data': 500, # train size (synth data)
			 'n_test': 1000, # test size (synth data)
			 'x_range': [0, 2], # x range to sample data from (synth data)
			 'activation': 'relu', # activation, relu or softplus
			 }

	if dataset_str in ['Gaussian_linear', 'Gaussian_nonlinear', 'Exponential', 'Weibull', 'Gaussian_uniform', 'LogNorm']:
		hyp['activation'] = 'relu'
		hyp['n_data'] = 500
		hyp['n_test'] = 1000
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 100
			
	if dataset_str in ['Norm_med','Norm_heavy','Norm_light','Norm_same']:
		hyp['n_data']=2000
		hyp['n_test']=1000
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20

	if dataset_str in ['LogNorm_med','LogNorm_heavy','LogNorm_light','LogNorm_same']:
		hyp['n_data']=4000
		hyp['n_test']=1000
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20

	if dataset_str in ['METABRIC']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	if dataset_str in ['WHAS']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 100
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass
	
	if dataset_str in ['SUPPORT']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = False
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass
	
	if dataset_str in ['GBSG']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass
	
	elif dataset_str in ['TMBImmuno']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass

	elif dataset_str in ['BreastMSK']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 100
			hyp['is_dropout'] = False
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 10
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = False
		elif loss_str == 'neocleous':
			pass

	if dataset_str in ['LGGGBM']:
		if loss_str == 'cqrnn':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'cqrnn_excl_censor':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'deepquantreg':
			hyp['n_epochs'] = 50
			hyp['is_dropout'] = True
		elif loss_str == 'lognorm':
			hyp['n_epochs'] = 20
			hyp['is_dropout'] = True
		elif loss_str == 'neocleous':
			pass
		
	return hyp