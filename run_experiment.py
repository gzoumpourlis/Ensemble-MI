import os
import sys
sys.dont_write_bytecode = True
import random
import datetime
import numpy as np
import pandas as pd

from src.config import *
from src.train import train
from src.args_utils import create_exp_args

random.seed(1312)

# Create the experiment's name string, containing date and time
now = datetime.datetime.now()
exp_name = now.strftime("%Y_%m_%d-%H%M%S")

exp_args_dict = {
    'use_folds': True,
    'n_folds': 5,
    'resume': False,
    'resume_exp_id': '',

    'use_subsets': True,
    'n_subsets': 7,
    'ensemble_loss_1': True,
    'ensemble_loss_2': True,
    'loss_2_single_expert': False,
    'n_modules_stage_1': 4,
    'stage_1_is_shared': False,
    'stage_2_is_shared': True,
    'feats_aggr': 'mean',

    'pre_align': True,
    'dataset_name': 'PhysionetMI',
    'batch_size': 64,
    'batch_size_val': 64,

    ############################
    # Network architecture
    'net_cls': 'ensemble', # eegnet | ensemble
    'backbone_k_width': 1,
    'dropout_input': 0.4,
    'dropout_1': 0.1,

    # Load pretrained checkpoint
    'load_ckpt': False,
    'ckpt_file': '',
    'checkpoint': 'best',

    # Use warmup period or scheduler for learning rate
    'use_scheduler': False,
    'warmup': False,
    ############################
    'cuda': True,
    'exp_group': exp_name,
    ############################
    # Training hyperparams
    'max_epochs': 120,
    'warmup_epochs': 0,
    'optim': 'sgd',
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 0.01,
    'coef_cls': 1.0,
}

def main(exp_args_dict):

    if exp_args_dict['use_subsets'] and (exp_args_dict['net_cls'] == 'eegnet'):
        print('EEGNet architecture does not support assigning subjects to subsets. Quitting...')
        quit()
    if (not exp_args_dict['use_subsets']) and (exp_args_dict['net_cls'] == 'ensemble'):
        print('EnsembleEEG architecture requires assigning subjects to subsets. Quitting...')
        quit()

    if exp_args_dict['use_folds']:
        fold_range = [fold_idx for fold_idx in range(exp_args_dict['n_folds'])]
        loop_range = fold_range
    else:
        subject_range = get_subject_list(exp_args_dict['dataset_name'])
        N_subs = len(subject_range)
        loop_range = subject_range

        if exp_args_dict['resume'] and (exp_args_dict['resume_exp_id']!=''):
            if exp_args_dict['use_folds']:
                eval_mode = '{}-fold'.format(exp_args_dict['n_folds'])

            else:
                eval_mode = 'LOSO'
            progress_folder = os.path.join(os.getcwd(), 'results', exp_args_dict['dataset_name'], eval_mode, exp_args_dict['resume_exp_id'])
            results_xls_filename = os.path.join(progress_folder, 'results.xlsx')
            if os.path.exists(results_xls_filename):
                df = pd.read_excel(results_xls_filename, index_col=0)
                subjects_done = df['Test subject'].values
                subject_range = [el for el in subject_range if el not in subjects_done]
                N_subs = len(subject_range)
                loop_range = subject_range

    cnt_exps = 0
    N_exps = len(loop_range)
    list_accs = list()
    for loop_iter_idx in loop_range:
        cnt_exps += 1

        if not exp_args_dict['use_folds']:
            args = create_exp_args(exp_args_dict=exp_args_dict, subject_idx=loop_iter_idx)
        else:
            args = create_exp_args(exp_args_dict=exp_args_dict, fold_idx=loop_iter_idx)
        acc, acc_by_loss, max_val_acc, min_val_loss = train(args)
        list_accs.append(acc)

        if not exp_args_dict['use_folds']:
            print('\n\nSubject {} ({}/{}) | Test accuracy: {:.2f}\n'.format(loop_iter_idx, cnt_exps, N_exps, acc))
        else:
            print('\n\nFold {} ({}/{}) | Test accuracy: {:.2f}\n'.format(loop_iter_idx + 1, loop_iter_idx + 1, N_exps, acc))

        print('Mean test accuracy: {:.2f}\n'.format(np.mean(list_accs)))

    mean_acc = np.mean(list_accs)
    std_acc = np.std(list_accs)
    if not exp_args_dict['use_folds']:
        print('\n\nMean test accuracy over {} subject(s): {:.2f} ± {:.2f}\n'.format(N_subs,
                                                                                    mean_acc,
                                                                                    std_acc))
    else:
        print('\n\nMean test accuracy over {} fold(s): {:.2f} ± {:.2f}\n'.format(exp_args_dict['n_folds'],
                                                                                 mean_acc,
                                                                                 std_acc))

if __name__ == '__main__':
    main(exp_args_dict)
