import os
import mne
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import *

def val(dataloader_val, net, criterion, history, record_history, log_preds, args, phase):

	if args.cuda:
		device = 'cuda'
	else:
		device = 'cpu'

	net.eval()

	if log_preds:
		subject_list = list()
		target_list = list()
		score_list = list()
		pred_list = list()

	for batch_idx, batch in enumerate(dataloader_val):
		inputs = batch['eeg'].float().to(device)
		targets = batch['target'].to(device)
		subjects = batch['subject']

		with torch.no_grad():
			if args.use_subsets:
				dataset_idx = batch['dataset_idx'].to(device)
				input_dict = {'x': inputs,
							  'dataset_idx': dataset_idx,
							  'training': False}
				out_dict = net(input_dict)
			else:
				out_dict = net(inputs)
			out = out_dict['scores']

			if args.use_subsets:
				for subset_idx in range(args.n_subsets):
					out_mean_idx = out[:, subset_idx, :]
					loss_cls_mean_idx = torch.nn.CrossEntropyLoss(reduction='none')(out_mean_idx, targets)
					loss_cls_mean_idx_w = 1.0 * loss_cls_mean_idx
					if subset_idx == 0:
						loss = loss_cls_mean_idx_w.mean()
					else:
						loss += loss_cls_mean_idx_w.mean()
			else:
				loss = criterion(out, targets)

		epoch_loss = loss.clone().detach().cpu().numpy()

		if args.use_subsets:
			# Mean over all subsets
			out_mean_moe = out.mean(1)
			_, predicted = out_mean_moe.max(1)
		else:
			_, predicted = out.max(1)

		if log_preds:
			for el in subjects:
				subject_list.append( int(el) )
			for el in targets:
				target_list.append( int(el.cpu().numpy()) )
			for el in out:
				score_list.append( el.cpu().numpy() )
			for el in predicted:
				pred_list.append( int(el.cpu().numpy()) )
		correct_batch = predicted.eq(targets).sum().item()

		if record_history:
			history.new_batch()
			history.record_batch('{}_loss'.format(phase), epoch_loss)
			history.record_batch('{}_acc'.format(phase), 100.0 * correct_batch / len(targets))
			history.record_batch('{}_batch_size'.format(phase), len(targets))
	if log_preds:
		subject_list = np.array(subject_list)
		target_list = np.array(target_list)
		score_list = np.array(score_list)
		pred_list = np.array(pred_list)

		subject_unique = dataloader_val.dataset.subject
		pred_dict = {'pred': pred_list,
					 'score': score_list,
					 'target': target_list,
					 'subject': subject_list,
					 'subject_unique': subject_unique
					 }
		acc_manual = np.round(100 * np.sum(target_list == pred_list) / len(pred_list), 2)
	else:
		acc_manual = 0.0
		pred_dict = {}
	return dataloader_val, net, criterion, history, args, acc_manual, pred_dict


def compute_loss_cls(out_dict_mean, targets, subset, n_subsets, device, coef_1):

	out_mean = out_dict_mean['scores']
	for subset_idx in range(n_subsets):
		subset_samples_mask = (subset.cpu().numpy() == subset_idx)
		subset_encoder_w = torch.zeros(size=(out_mean.shape[0],)).float().to(device)
		if np.any(subset_samples_mask):
			subset_samples_inds = np.where(subset_samples_mask)[0]
			subset_encoder_w[subset_samples_inds] += 1.0
		out_mean_idx = out_mean[:, subset_idx, :]
		loss_cls_mean_idx = torch.nn.CrossEntropyLoss(reduction='none')(out_mean_idx, targets)
		coef_w = (coef_1 + subset_encoder_w)
		coef_w = coef_w.clamp(max=1.0)
		loss_cls_mean_idx_w = coef_w * loss_cls_mean_idx
		if subset_idx == 0:
			loss_cls_mean_w = loss_cls_mean_idx_w.mean()
		else:
			loss_cls_mean_w += loss_cls_mean_idx_w.mean()
	return loss_cls_mean_w


def compute_loss_cls_distill(out_dict_mean, subset, n_subsets, device, single_expert=False):

	subset_indices = [el for el in range(n_subsets)]

	out_mean = out_dict_mean['scores']
	for subset_idx in range(n_subsets):
		if not single_expert:
			# k-1 experts
			indices_not_idx = [el for el in subset_indices if el != subset_idx]
			pseudo_targets = out_mean[:, indices_not_idx, :].detach()
			pseudo_targets = pseudo_targets.mean(1)
		else:
			# single expert
			pseudo_targets = torch.zeros(size=(out_mean.shape[0], out_mean.shape[2])).float().to(device)
			subset_npy = subset.cpu().numpy()
			for batch_i_cnt in range(out_mean.shape[0]):
				subset_i_index = subset_npy[batch_i_cnt]
				pseudo_targets[batch_i_cnt] = out_mean[batch_i_cnt, subset_i_index]
			pseudo_targets = pseudo_targets.detach()
		pseudo_targets = torch.nn.Softmax(dim=1)(pseudo_targets)
		subset_samples_mask = (subset.cpu().numpy() != subset_idx)
		subset_encoder_w = torch.from_numpy(1.0 * subset_samples_mask).float().to(device)
		out_mean_idx = out_mean[:, subset_idx, :]
		loss_cls_mean_idx = torch.nn.CrossEntropyLoss(reduction='none')(out_mean_idx, pseudo_targets)
		# mask samples belonging to k-th subset
		loss_cls_mean_idx_w = subset_encoder_w * loss_cls_mean_idx
		# TODO: check, do we need any coefficient, to fix issue of subset zeros?
		if subset_idx == 0:
			loss_cls_mean_w_distill = loss_cls_mean_idx_w.mean()
		else:
			loss_cls_mean_w_distill += loss_cls_mean_idx_w.mean()

	return loss_cls_mean_w_distill

def compute_weighted_preds(out_dict_mean):
	out_mean = out_dict_mean['scores']
	out_mean_moe = out_mean.mean(1)
	return out_mean_moe

def get_input_dict(batch, device):

	inputs = batch['eeg'].float().to(device)
	subset = batch['subset']
	dataset_idx = batch['dataset_idx'].to(device)
	input_dict = {'x': inputs,
				'subset': subset,
				'dataset_idx': dataset_idx}

	return input_dict

def get_picked_accs(history):

	results_columns = ['train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc']
	df = pd.DataFrame(history[:, results_columns], columns=results_columns, index=history[:, 'epoch'])
	val_losses = df['val_loss'].values
	val_accs = df['val_acc'].values
	test_accs = df['test_acc'].values
	min_val_loss = np.min(val_losses)
	max_val_acc = np.max(val_accs)
	max_test_acc = np.max(test_accs)
	min_val_loss_index = np.argmin(val_losses)
	max_val_acc_index = np.argmax(val_accs)
	picked_test_acc = test_accs[max_val_acc_index]
	picked_test_acc_by_loss = test_accs[min_val_loss_index]

	return min_val_loss, max_val_acc, max_test_acc, picked_test_acc, picked_test_acc_by_loss

def update_history_on_phase_end(history, phase, get_acc=False, acc=None):

	batch_history = history[-1]['batches']
	batch_sizes = np.array([el['{}_batch_size'.format(phase)] for el in batch_history if '{}_batch_size'.format(phase) in el.keys()])
	batch_losses = np.array([el['{}_loss'.format(phase)] for el in batch_history if '{}_loss'.format(phase) in el.keys()])
	batch_accs = np.array([el['{}_acc'.format(phase)] for el in batch_history if '{}_acc'.format(phase) in el.keys()])

	epoch_phase_loss = np.sum(batch_sizes * batch_losses) / np.sum(batch_sizes)
	if get_acc:
		epoch_phase_acc = acc
	else:
		epoch_phase_acc = np.round( np.sum(batch_sizes * batch_accs) / np.sum(batch_sizes), 2)
	history.record('{}_loss'.format(phase), epoch_phase_loss)
	history.record('{}_acc'.format(phase), epoch_phase_acc)

	phase_losses = np.array([el['{}_loss'.format(phase)] for el in history[:]])
	phase_accs = np.array([el['{}_acc'.format(phase)] for el in history[:]])
	if epoch_phase_loss == np.min(phase_losses):
		history.record('{}_loss_best'.format(phase), True)
	if epoch_phase_acc == np.max(phase_accs):
		history.record('{}_acc_best'.format(phase), True)

	return history

def plot_filters(net, args, fold_cnt):

	ch_names = get_channel_list(args.dataset_name)
	C = len(ch_names)  # Change this to the number of electrodes in your montage
	# Create an EEG montage
	montage = mne.channels.make_standard_montage('standard_1020')
	# Create an empty Info object
	info = mne.create_info(ch_names=ch_names, sfreq=args.sfreq, ch_types='eeg')
	raw_data = np.random.rand(C, args.sfreq)
	raw = mne.io.RawArray(raw_data, info)
	raw.set_montage(montage)

	if args.net_cls == 'eegnet':
		filters = net.conv_spatial.weight.detach().squeeze().cpu().numpy()
		N_filters = filters.shape[0]
		N_rows = int( np.ceil(np.sqrt(N_filters)) )
		N_cols = N_rows
		# Create a figure with subplots
		fig, axes = plt.subplots(N_rows, N_cols, figsize=(10, 10))  # Adjust figsize as needed
		max_abs = np.max(np.abs(filters))
		for i in range(N_rows):
			for j in range(N_cols):
				ax = axes[i, j]
				ax.set_title('Filter #{:02d}'.format(i * N_rows + j + 1))  # Adjust the title as needed
				mne.viz.plot_topomap(filters[i * N_rows + j], raw.info, axes=ax, show=False, cnorm=None,
									 vlim=(-1.01 * max_abs, 1.01 * max_abs))
				fig.colorbar(ax.get_images()[0], ax=ax)

		# Add spacing between subplots
		plt.tight_layout()
		folder_path = os.path.join(os.getcwd(), 'plots_topo')
		if not os.path.exists(folder_path):
			os.makedirs(folder_path, exist_ok=True)
		filename = '{}_fold-{:02d}'.format(args.exp_group, fold_cnt)
		filename_svg = os.path.join(folder_path, '{}.svg'.format(filename))
		plt.savefig(filename_svg)
		filename_eps = os.path.join(folder_path, '{}.eps'.format(filename))
		plt.savefig(filename_eps)
		# Show the figure
		# plt.show()
		# Close the figure
		plt.close()
	elif args.net_cls == 'ensemble':
		list_filters = list()
		for subset_cnt in range(args.n_subsets):
			list_filters.append(net.stage_1.encoders['encoder_{:03d}'.format(subset_cnt + 1)].modules_all.module_1[0].weight.detach().squeeze().cpu().numpy())
		N_filters = list_filters[0].shape[0]
		N_rows = int( np.ceil(np.sqrt(N_filters)) )
		N_cols = N_rows

		for subset_cnt in range(args.n_subsets):
			filters = list_filters[subset_cnt]
			# Create a figure with subplots
			fig, axes = plt.subplots(N_rows, N_cols, figsize=(10, 10))  # Adjust figsize as needed
			max_abs = np.max(np.abs(filters))
			for i in range(N_rows):
				for j in range(N_cols):
					ax = axes[i, j]
					ax.set_title('Filter #{:02d}'.format(i * N_rows + j + 1))  # Adjust the title as needed
					mne.viz.plot_topomap(filters[i * N_rows + j], raw.info, axes=ax, show=False, cnorm=None,
										 vlim=(-1.01 * max_abs, 1.01 * max_abs))
					fig.colorbar(ax.get_images()[0], ax=ax)

			# Add spacing between subplots
			plt.tight_layout()
			folder_path = os.path.join(os.getcwd(), 'plots_topo')
			if not os.path.exists(folder_path):
				os.makedirs(folder_path, exist_ok=True)
			filename = '{}_fold-{:02d}_subnetwork-{:02d}'.format(args.exp_group, fold_cnt, subset_cnt + 1)
			filename_svg = os.path.join(folder_path, '{}.svg'.format(filename))
			plt.savefig(filename_svg)
			filename_eps = os.path.join(folder_path, '{}.eps'.format(filename))
			plt.savefig(filename_eps)
			# Show the figure
			# plt.show()
			# Close the figure
			plt.close()
