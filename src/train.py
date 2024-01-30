import os
import sys
import torch
import numpy as np
import pandas as pd
from braindecode.util import set_random_seeds

from .utils import *
from .initializers import *

def train(args):

	# Checking CUDA availability
	if args.cuda and torch.cuda.is_available():
		torch.backends.cudnn.benchmark = True
		device = 'cuda'
	else:
		device = 'cpu'

	# random seed to make results reproducible
	seed = 20200220
	set_random_seeds(seed=seed, cuda=args.cuda)

	if args.use_folds:
		eval_mode = '{}-fold'.format(args.n_folds)

	else:
		eval_mode = 'LOSO'
	progress_folder = os.path.join(os.getcwd(), 'results', args.dataset_name, eval_mode, args.exp_group)
	if not os.path.exists(progress_folder):
		os.makedirs(progress_folder, exist_ok=True)

	args.progress_folder = progress_folder

	############################
	# Network is randomly initialized, no pretrained checkpoint is used
	dataloader_train, dataloader_val, dataloader_test, dataset_info = init_dataset(args=args)
	############################
	# Net initialization

	net = init_net(args, dataset_info)
	# Loading checkpoint of a pretrained model
	if args.load_ckpt:
		ckpt_filename = os.path.join(os.getcwd(), args.ckpt_file)
		net.load_state_dict(torch.load(ckpt_filename), strict=False)
		print('\nLoaded pretrained checkpoint: {}'.format(ckpt_filename))

	############################
	# Initialize criteria and optimizers/schedulers

	# Criterion for classification loss
	criterion = init_criterion()

	model_parameters = filter(lambda p: p.requires_grad, net.parameters())
	params = sum([np.prod(p.size()) for p in model_parameters])
	print('Model parameters: {}'.format(params))

	optimizer = init_optimizer(args, net)
	############################


	# History object, used to keep/print epoch statistics
	print_log = init_print_log()
	history = init_history()
	# Initialize best val/test accuracy, to be used for checkpoint saving
	best_val_acc = 0

	print('\nTraining...\n')
	############################

	for epoch in range(args.max_epochs):
		if epoch==60:
			for param_group in optimizer.param_groups:
				param_group['lr'] = 0.002
		history.new_epoch()
		history.record('epoch', len(history))
		############################
		# Set networks to training mode
		net.train()
		# Initialize number of correctly classified training samples, total training samples, and batches
		epoch_start_t = time.time()
		# Iterate over the training set
		for batch_idx, batch in enumerate(dataloader_train):
			if batch['eeg'].shape[0]<args.batch_size:
				continue
			targets = batch['target'].to(device)
			if args.use_subsets:
				subset = batch['subset']
				coef_alpha = 1.0 - (epoch / args.max_epochs)
				coef_1 = coef_alpha
				coef_2 = 1.0 - coef_alpha
				input_dict_mean = get_input_dict(batch, device)
				out_dict_mean = net(input_dict_mean)
				loss_cls_mean_w = compute_loss_cls(out_dict_mean,
												   targets,
												   subset,
												   args.n_subsets,
												   device,
												   coef_1)
				loss_cls_mean_w_distill = compute_loss_cls_distill(out_dict_mean,
																   subset,
																   args.n_subsets,
																   device,
																   single_expert=args.loss_2_single_expert)
				term_1 = args.n_subsets * loss_cls_mean_w
				term_2 = 0.7 * coef_2 * loss_cls_mean_w_distill
				if args.ensemble_loss_1 and args.ensemble_loss_1:
					loss = term_1 + term_2
				elif args.ensemble_loss_1:
					loss = term_1
			else:
				# Input shape: Batch x 1 x Channels x Time_length
				inputs = batch['eeg'].float().to(device)
				out_dict = net(inputs)
				out = out_dict['scores']
				loss_cls = criterion(out, targets)
				loss = args.coef_cls * loss_cls
			############################
			# Do backward pass & optimizer step
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=0.10)
			optimizer.step()
			############################
			# Keep batch statistics (loss, acc)
			batch_loss = loss.item()
			del loss
			# Predicted class labels for training samples
			if args.use_subsets:
				out_ = compute_weighted_preds(out_dict_mean)
			else:
				out_ = out
			_, predicted = out_.max(1)
			batch_correct = predicted.eq(targets).sum().item()
			batch_acc = 100.0 * batch_correct / targets.size(0)
			# Logging train batch statistics on history
			history.new_batch()
			history.record_batch('train_loss', batch_loss)
			history.record_batch('train_acc', batch_acc)
			history.record_batch('train_batch_size', len(targets))

		# Logging train epoch statistics on history
		history = update_history_on_phase_end(history=history, phase='train')
		############################
		# Validation phase
		dataloader_val, net, criterion, history, args, epoch_val_acc, pred_dict_val = val(dataloader_val,
																						  net,
																						  criterion,
																						  history,
																						  record_history=True,
																						  log_preds=True,
																						  args=args,
																						  phase='val')
		# Logging validation epoch statistics on history
		history = update_history_on_phase_end(history=history, phase='val', get_acc=True, acc=epoch_val_acc)
		# Test phase
		dataloader_test, net, criterion, history, args, epoch_test_acc, pred_dict_test = val(dataloader_test,
																							 net,
																							 criterion,
																							 history,
																							 record_history=True,
																							 log_preds=True,
																							 args=args,
																							 phase='test')
		# Logging test epoch statistics on history
		history = update_history_on_phase_end(history=history, phase='test', get_acc=True, acc=epoch_test_acc)
		epoch_end_t = time.time()
		history.record('dur', epoch_end_t - epoch_start_t)
		# Print accuracy/loss for train/val phase, on epoch end
		print_log.on_epoch_end(history=history, verbose=True)
		############################

		# In case of new max val. accuracy, save model checkpoint
		if epoch_val_acc > best_val_acc:
			best_val_acc = epoch_val_acc
			root_path = os.path.join(os.getcwd(), 'checkpoints')
			if not os.path.exists(root_path):
				os.makedirs(root_path, exist_ok=True)
			if args.use_folds:
				model_best_filename = os.path.join(root_path, 'net_best_{}_fold_{:02d}.pth'.format(args.exp_group, args.fold))
			else:
				model_best_filename = os.path.join(root_path, 'net_best_{}_subj_{:03d}.pth'.format(args.exp_group,
																								 args.subject))
			torch.save(net.state_dict(), model_best_filename)

	#########################################################

	net.load_state_dict(torch.load(model_best_filename), strict=False)
	plot_filters(net=net, args=args, fold_cnt=args.fold + 1)

	#########################################################

	min_val_loss, max_val_acc, max_test_acc, picked_test_acc, picked_test_acc_by_loss = get_picked_accs(history)
	results_xls_filename = os.path.join(args.progress_folder, 'results.xlsx')
	if os.path.exists(results_xls_filename):
		df_previous = pd.read_excel(results_xls_filename, index_col=0)
		existing = True
	else:
		existing = False

	if not args.use_folds:
		column_names = ['Test subject',]
	else:
		column_names = ['Test fold', ]
	column_names.append('Test acc.')
	column_names.append('Max Test acc.')
	results_list = list()
	if not args.use_folds:
		results_data = [args.subject, ]
	else:
		results_data = [args.fold + 1, ]
	results_data.append(picked_test_acc)
	results_data.append(max_test_acc)
	results_list.append(results_data)
	df = pd.DataFrame(results_list, columns=column_names)

	if existing:
		df = df_previous.append(df, ignore_index=True)

	df.to_excel(results_xls_filename)
	############################

	return picked_test_acc, picked_test_acc_by_loss, max_val_acc, min_val_loss