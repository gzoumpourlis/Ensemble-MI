import os
import json
import copy
import argparse

def parse_default_args():
	##########################################################################
	parser = argparse.ArgumentParser(description="Beetl Competition")
	parser.add_argument("--exp_group", default='',
						help="name of experiment folder, when executing a group of experiments")
	parser.add_argument("--comment", default='', help="comment to append on folder name")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument('--dataset_name', type=str, choices=('PhysionetMI'), default='PhysionetMI',
						help="set training set name")
	parser.add_argument('--mode', type=str, choices=('intra_subject', 'cross_session', 'cross_subject', 'cross_dataset'), default='intra_subject',
						help="set training mode")

	parser.add_argument('--fmin', type=float, default=4., help="set min frequency for bandpass filtering")
	parser.add_argument('--fmax', type=float, default=38., help="set max frequency for bandpass filtering")
	parser.add_argument('--sfreq', type=int, default=100, help="target frequency")
	parser.add_argument('--t_start_offset', type=float, default=0.0, help="trial start offset")
	parser.add_argument('--t_end_offset', type=float, default=0.0, help="trial start offset")
	parser.add_argument('--t_dur', type=float, default=0.0, help="trial duration")

	parser.add_argument('--subject', type=int, default=1, help="in case of LOSO, set ID of the only test subject") # 1 to N
	parser.add_argument('--fold', type=int, default=0, help="in case of k-fold cross validation, set fold number") # 0 to N-1
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument("--net_cls", default='eegnet', choices=('eegnet', 'ensemble'), help="name of model")
	parser.add_argument('--backbone_k_width', type=int, default=1, help="backbone net width factor")
	parser.add_argument("--coef_cls", type=float, default=1.0, help="coefficient of classification loss")
	parser.add_argument('--checkpoint', type=str, choices=('best', 'last'), default='best', help="model ckpt")
	parser.add_argument('--load_ckpt', dest='load_ckpt', action='store_true', help="load ckpt of pretrained model")
	parser.set_defaults(load_ckpt=False)
	parser.add_argument('--ckpt_file', type=str, default='', help="model ckpt file")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument('--max_epochs', type=int, default=120, help="max number of epochs")
	parser.add_argument('--warmup_epochs', type=int, default=0, help="number of warmup epochs")
	parser.add_argument('--optim', type=str, default='sgd', choices=('adam', 'adamw', 'sgd'), help="optimizer")
	parser.add_argument('--use_scheduler', dest='use_scheduler', action='store_true', help="use optim scheduler")
	parser.set_defaults(use_scheduler=False)
	parser.add_argument('--warmup', dest='warmup', action='store_true', help="use lr warmup")
	parser.set_defaults(warmup=False)
	parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
	parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
	parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay")
	parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
	parser.add_argument('--batch_size_val', type=int, default=1, help="batch size for validation/test")
	parser.add_argument('--dropout_input', type=float, default=0.0, help="value for dropout applied at the input of the network")
	parser.add_argument('--dropout_1', type=float, default=0.0, help="value for dropout applied at layer dropout_1")
	parser.add_argument('--n_folds', type=int, default=5, help="number of folds for LMSO/k-fold")
	# ---------------------------------------------------------------------------------------------------------------- #
	parser.add_argument('--use_subsets', dest='use_subsets', action='store_true', help="")
	parser.set_defaults(use_subsets=False)
	parser.add_argument('--n_subsets', type=int, default=2, help="number of subsets")
	parser.add_argument('--ensemble_loss_1', dest='ensemble_loss_1', action='store_true', help="ensemble loss 1 (curriculum)")
	parser.set_defaults(ensemble_loss_1=False)
	parser.add_argument('--ensemble_loss_2', dest='ensemble_loss_2', action='store_true', help="ensemble loss 2 (distillation)")
	parser.set_defaults(ensemble_loss_2=False)
	parser.add_argument('--loss_2_single_expert', dest='loss_2_single_expert', action='store_true', help="ensemble loss 2 distill, single expert or not")
	parser.set_defaults(loss_2_single_expert=False)

	parser.add_argument('--n_modules_stage_1', type=int, default=4, help="number of stage_1 layers")
	parser.add_argument('--stage_1_is_shared', dest='stage_1_is_shared', action='store_true', help="stage 1")
	parser.set_defaults(stage_1_is_shared=False)
	parser.add_argument('--stage_2_is_shared', dest='stage_2_is_shared', action='store_true', help="stage 2")
	parser.set_defaults(stage_2_is_shared=False)
	parser.add_argument('--feats_aggr', type=str, default='mean', choices=('mean', 'concat'), help="")

	parser.add_argument('--use_folds', dest='use_folds', action='store_true',
						help="k-fold or Leave-One-Subject-Out")
	parser.set_defaults(use_folds=False)
	parser.add_argument('--use_car', dest='use_car', action='store_true', help="perform Common Average Reference on EEG signals")
	parser.set_defaults(use_car=False)
	parser.add_argument('--pre_align', dest='pre_align', action='store_true', help="perform EEG alignment using covariances")
	parser.set_defaults(pre_align=False)
	parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA")
	parser.set_defaults(cuda=False)
	######################################################################
	default_args = parser.parse_args()

	return default_args

def overwrite_default_args(exp_args, default_args):

	for arg_key in vars(exp_args):
		arg_val = getattr(exp_args, arg_key)
		setattr(default_args, arg_key, arg_val)

	args = copy.deepcopy(default_args)

	return args

def dict_to_namespace(args):
	
	# Write arguments from dictionary to JSON
	json_path = write_args_json(args)

	# Read arguments from JSON to Namespace
	parser = argparse.ArgumentParser()
	with open(json_path, 'rt') as f:
		args_namespace = argparse.Namespace()
		args_namespace.__dict__.update(json.load(f))
		exp_args = parser.parse_args(namespace=args_namespace)

	return exp_args

def write_args_json(args):

	json_folder = os.path.join(os.getcwd(), 'json')
	if not os.path.exists(json_folder):
		os.makedirs(json_folder, exist_ok=True)

	filename = 'exp_args_{}.json'.format(args['exp_group'])
	json_path = os.path.join(json_folder, filename)

	with open(json_path, 'w') as out:
		json.dump(args, out)

	return json_path

def create_exp_args(exp_args_dict, subject_idx=-1, fold_idx=-1):

	# default args
	default_args_namespace = parse_default_args()

	# exp args
	exp_args_dict.update({'subject': subject_idx})
	exp_args_dict.update({'fold': fold_idx})
	exp_args_namespace = dict_to_namespace(exp_args_dict)

	# overwrite default with exp args
	args = overwrite_default_args(exp_args_namespace, default_args_namespace)

	return args