import os
import sys
import mne
mne.set_log_level('CRITICAL')
import time
import copy
import random
import pickle
import numpy as np
from collections import deque

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing.preprocess import (preprocess, Preprocessor)

from models import *
from third_party.skorch import PrintLog, History
from .config import *
from .custom_dataset import CustomDataset

def init_dataset(args):

    dataset_name = args.dataset_name
    print('Training on {} dataset'.format(dataset_name))

    # Using dataset details from src/config.py
    subject_range = get_subject_list(dataset_name)
    dataset_targets = get_target_list(dataset_name)
    dataset_events = get_event_list(dataset_name)
    dataset_channels = get_channel_list(dataset_name)

    ############################
    # Load dataset

    ds_list = list()
    for subj in subject_range:
        dataset_root_path = os.path.join(os.getcwd(),
                                         'preprocessed_data',
                                         dataset_name)
        dataset_filename = 'windows_dataset_{:03}.pkl'.format(subj)
        dataset_filepath = os.path.join(dataset_root_path, dataset_filename)
        with open(dataset_filepath, 'rb') as inp:
            windows_ds = pickle.load(inp)
        ds_list.append(windows_ds)
    windows_dataset = BaseConcatDataset(ds_list)

    # Optionally, keep a subset of the existing electrodes
    preprocessors = [Preprocessor('pick_channels', ch_names=dataset_channels, ordered=True)]
    print('\nPreprocessing pre-loaded dataset')
    preprocess(windows_dataset, preprocessors)

    ############################
    # Split dataset

    subject_column = windows_dataset.description['subject'].values
    if not args.use_folds:

        inds_test = list(np.where(subject_column == args.subject)[0])
        subjects_trainval = list()
        inds_train = list()
        inds_val = list()
        for sub_idx, sub in enumerate(subject_column):
            if sub_idx not in inds_test:
                subjects_trainval.append(sub)
        subjects_trainval_unique = list(np.unique(subjects_trainval))
        subjects_trainval_unique_shuffled = copy.deepcopy(subjects_trainval_unique)
        random.shuffle(subjects_trainval_unique_shuffled)

        N_subjects_trainval = len(subjects_trainval_unique)
        N_subjects_train = int(0.8*N_subjects_trainval)
        subjects_train = subjects_trainval_unique_shuffled[:N_subjects_train]
        subjects_val = subjects_trainval_unique_shuffled[N_subjects_train:]
        subjects_test = [args.subject, ]

        common_train_val = set(subjects_train).intersection(subjects_val)
        common_train_test = set(subjects_train).intersection(subjects_test)
        common_val_test = set(subjects_val).intersection(subjects_test)
        assert (len(common_train_val) == 0)
        assert (len(common_train_test) == 0)
        assert (len(common_val_test) == 0)

        for sub_idx, sub in enumerate(subject_column):
            if sub in subjects_train:
                inds_train.append(sub_idx)
            elif sub in subjects_val:
                inds_val.append(sub_idx)

        print('Train subjects (N={:03d}) : {}'.format(len(subjects_train), subjects_train))
        print('Val   subjects (N={:03d}) : {}'.format(len(subjects_val), subjects_val))
        print('Test  subjects (N={:03d}) : {}'.format(len(subjects_test), subjects_test))

        # Save file containing subject splitting
        split_dict = {'train': subjects_train,
                     'val': subjects_val,
                     'test': subjects_test}
        split_filename = os.path.join(args.progress_folder, 'split_subject_{:03d}.pkl'.format(args.subject))
        with open(split_filename, 'wb') as outp:
            pickle.dump(split_dict, outp, pickle.HIGHEST_PROTOCOL)

    else:
        splits = np.array_split(subject_range, args.n_folds)
        splits = deque(splits)
        splits.rotate(args.fold + 1)
        split_test = splits[0]
        split_val = splits[1]
        splits_train = [splits[i] for i in range(2, args.n_folds)]
        split_train = np.hstack(splits_train)

        subjects_train = [el for el in split_train]
        subjects_val = [el for el in split_val]
        subjects_test = [el for el in split_test]

        print('Train subjects (N={:03d}) : {}'.format(len(subjects_train), subjects_train))
        print('Val   subjects (N={:03d}) : {}'.format(len(subjects_val), subjects_val))
        print('Test  subjects (N={:03d}) : {}'.format(len(subjects_test), subjects_test))

        # Save file containing subject splitting
        split_dict = {'train': subjects_train,
                      'val': subjects_val,
                      'test': subjects_test}
        split_filename = os.path.join(args.progress_folder, 'split_fold_{:03d}.pkl'.format(args.fold + 1))
        with open(split_filename, 'wb') as outp:
            pickle.dump(split_dict, outp, pickle.HIGHEST_PROTOCOL)
        ############################

        inds_train = list()
        inds_val = list()
        inds_test = list()
        for sub_idx, sub in enumerate(subject_column):
            if sub in subjects_train:
                inds_train.append(sub_idx)
            elif sub in subjects_val:
                inds_val.append(sub_idx)
            elif sub in subjects_test:
                inds_test.append(sub_idx)

    splitted = windows_dataset.split([inds_train, inds_val, inds_test])
    train_set = splitted['0']
    val_set = splitted['1']
    test_set = splitted['2']

    # Merge multiple datasets into a single WindowDataset
    print('\nCreating custom dataset')
    t1 = time.time()

    dataset_train = CustomDataset(windows_dataset=train_set,
                                  phase='train',
                                  dataset_name=dataset_name,
                                  targets=dataset_targets,
                                  events=dataset_events,
                                  use_subsets=args.use_subsets,
                                  n_subsets=args.n_subsets)
    dataset_val = CustomDataset(windows_dataset=val_set,
                                phase='val',
                                dataset_name=dataset_name,
                                targets=dataset_targets,
                                events=dataset_events)
    dataset_test = CustomDataset(windows_dataset=test_set,
                                 phase='test',
                                 dataset_name=dataset_name,
                                 targets=dataset_targets,
                                 events=dataset_events)

    # Align EEG signals only once, before starting the train/val process
    if args.pre_align:
        dataset_train.covariance_align()
        dataset_val.covariance_align()
        dataset_test.covariance_align()

    t2 = time.time()
    print('Time elapsed: {:.2f} seconds'.format(t2 - t1))

    # Create dataloaders
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=0)
    dataloader_val = DataLoader(dataset_val,
                                batch_size=args.batch_size_val,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=0)
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=0)

    dataset_info = {'N_datasets': dataset_train.N_datasets,
                    'N_electrodes': dataset_train.N_electrodes,
                    'N_classes': dataset_train.N_classes,
                    'N_trials': dataset_train.N_trials,
                    'window_size': dataset_train.window_size}
    if args.use_subsets:
        dataset_info['N_subsets'] = dataset_train.n_subsets

    print('\n{} dataset | Dataset info:'.format(dataset_name))
    print(dataset_info)
    print('Samples | Train: {} | Val: {} | Test: {}'.format(dataset_train.N_trials,
                                                            dataset_val.N_trials,
                                                            dataset_test.N_trials))

    return dataloader_train, dataloader_val, dataloader_test, dataset_info

def init_net(args, dataset_info):

    if args.net_cls == 'eegnet':
        net = EEGNetv4_class(in_chans=dataset_info['N_electrodes'],
                             n_classes=dataset_info['N_classes'],
                             input_window_samples=dataset_info['window_size'],
                             final_conv_length="auto",
                             pool_mode="max",
                             F1=args.backbone_k_width * 8,
                             D=2,
                             F2=args.backbone_k_width * 16,
                             kernel_length=64,
                             dropout_input_prob=args.dropout_input,
                             dropout_1_prob=args.dropout_1)
    elif args.net_cls == 'ensemble':
        net = EnsembleEEG(n_datasets_train=dataset_info['N_datasets'],
                          in_chans=dataset_info['N_electrodes'],
                          input_window_samples=dataset_info['window_size'],
                          dropout_input_prob=args.dropout_input,
                          dropout_1_prob=args.dropout_1,
                          n_subsets=dataset_info['N_subsets'],
                          k_width=args.backbone_k_width,
                          n_classes=dataset_info['N_classes'],
                          n_modules_backbone=args.n_modules_stage_1,
                          stage_1_is_shared=args.stage_1_is_shared,
                          stage_2_is_shared=args.stage_2_is_shared)
    else:
        print('Network architecture not implemented yet. Quitting...')
        quit()

    if args.cuda:
        net.cuda()

    return net

def init_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion

def init_optimizer(args, net):

    if args.use_subsets:
        parameters = list( net.stage_1.parameters() ) + list( net.stage_2.parameters() )
    else:
        parameters = net.parameters()

    if args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

def init_print_log():
    print_log = PrintLog().initialize()
    return print_log

def init_history():
    history = History()
    return history