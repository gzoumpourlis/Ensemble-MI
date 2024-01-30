import os
import pdb
import sys
import mne
import torch
import random
import pandas as pd
import numpy as np
from numpy.random import shuffle
from torch.utils.data import Dataset

from scipy.linalg import pinv, sqrtm
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann

class CustomDataset(Dataset):

    def __init__(self,
                 windows_dataset,
                 phase=None,
                 dataset_name=None,
                 targets=None,
                 events=None,
                 use_subsets=False,
                 n_subsets=None,
                 ):
        """
        Args:
            phase (string): The phase (training, validation/test) that the dataset is used.
            dataset_name (string): Dataset name
            targets (list): List containing the names of the target classes
            events (dictionary): Dictionary with the correspondence class_name-class_ID
        """
        self.targets = targets
        self.N_classes = len(self.targets)
        self.targets_rearranged = [el for el in range(self.N_classes)]
        self.events_original = events.copy()

        self.phase = phase
        self.dataset_name = dataset_name

        self.use_subsets = use_subsets
        if self.use_subsets:
            self.n_subsets = n_subsets

        ###########################

        self.events_rearranged = {}
        self.label_correspondence = {}
        for target_name, target_num in zip(self.targets, self.targets_rearranged):
            self.events_rearranged.update( {target_name: target_num} )
            self.label_correspondence.update({events[target_name]: target_num}) # old_label --> new_label

        # Subsample events, using only the kept targets (when discarding some classes from a dataset)
        self.events = {}
        self.event_ids = []
        for target in self.targets:
            self.events.update( {target: events[target]} )
            self.event_ids.append(events[target])

        ###################################################
        list_subject = []
        list_session = []
        list_run = []
        list_metadata = []
        for ds in windows_dataset.datasets:
            sub = ds.description['subject']
            sess = ds.description['session']
            run = ds.description['run']
            metadata = ds.metadata
            list_subject.append(sub)
            list_session.append(sess)
            list_run.append(run)
            list_metadata.append(metadata)

        ###################################################
        # Merge runs per session

        list_merged_subject = []
        list_merged_session = []
        list_merged_metadata = []
        list_merged_epochs = []
        list_merged_ntrials = []

        # This list will be used to uniquely identify each sample of the dataset, with a pair of indices:
        # sub-dataset index, and trial index
        list_sample_indices = []

        unique_subjects = []
        unique_sessions = []
        for sub in list_subject:
            if sub not in unique_subjects:
                unique_subjects.append(sub)
        for sess in list_session:
            if sess not in unique_sessions:
                unique_sessions.append(sess)

        cnt_trials = 0
        dataset_index = 0
        for sub in unique_subjects:
            indices_sub = [i for i, x in enumerate(list_subject) if x == sub]
            for sess in unique_sessions:
                indices_sess = [i for i, x in enumerate(list_session) if x == sess]
                # sub/sess intersection
                indices_sub_sess = intersection(indices_sub, indices_sess)

                if not indices_sub_sess:
                    # empty list
                    continue

                # Concatenate metadata and epochs from multiple runs
                metadata = pd.concat([windows_dataset.datasets[index].metadata for index in indices_sub_sess])

                info = mne.create_info(ch_names=windows_dataset.datasets[0].raw.info['ch_names'],
                                       sfreq=windows_dataset.datasets[0].raw.info['sfreq'],
                                       ch_types='eeg')
                epochs_all = list()
                for index_dataset in indices_sub_sess:
                    epochs_dataset = list()
                    labels_dataset = list()
                    N_dataset_events = windows_dataset.datasets[index_dataset].__len__()
                    for index_event in range(N_dataset_events):
                        window_, y, crop_inds = windows_dataset.datasets[index_dataset].__getitem__(index=index_event)
                        epochs_dataset.append(window_)
                        labels_dataset.append(y)
                    epochs_dataset = np.array(epochs_dataset)
                    labels_dataset = np.array(labels_dataset)
                    events_from_annot, event_dict = mne.events_from_annotations(windows_dataset.datasets[index_dataset].raw)
                    events_from_annot_orig = events_from_annot.copy()
                    event_dict_orig = event_dict.copy()
                    for event_name, event_id in event_dict_orig.items():
                        inds_ = np.where(events_from_annot_orig[:,2] == event_id)[0]
                        events_from_annot[inds_,2] = self.events_original[event_name]
                        event_dict[event_name] = self.events_original[event_name]

                    epochs_dataset = mne.EpochsArray(epochs_dataset,
                                                     info,
                                                     events=events_from_annot,
                                                     event_id=event_dict)
                    epochs_all.append(epochs_dataset)
                epochs = mne.concatenate_epochs(epochs_all)

                subject_column = [sub for row_cnt in range(len(metadata))]
                session_column = [sess for row_cnt in range(len(metadata))]
                metadata['subject'] = subject_column
                metadata['session'] = session_column

                ####################################
                # Subsample trials, belonging in a subset of classes (picking *some* samples per class)
                # This is done to ensure class balance, by undersampling classes that have many samples
                if self.dataset_name == 'PhysionetMI':
                    trials_per_class = list()
                    for event_id, target in zip(self.event_ids, self.targets):
                        trials_per_class.append(len(epochs[target]))
                    # number of samples for each class
                    trials_per_class = np.array(trials_per_class)
                    # cardinality of each class after subsampling
                    min_n_trials_per_class = np.min(trials_per_class)

                    vec_targets = metadata['target'].values
                    condition_array = np.full((len(vec_targets)), False, dtype=bool)
                    for event_id in self.event_ids:
                        inds_event = np.where(vec_targets == event_id)[0]
                        if self.phase=='train':
                            # keeping balanced samples across classes
                            condition_event = np.full((len(vec_targets)), False, dtype=bool)
                            inds_event_kept = random.sample(list(inds_event), min_n_trials_per_class)
                            condition_event[inds_event_kept] = True
                        else:
                            # keeping all samples
                            condition_event = np.full((len(vec_targets)), False, dtype=bool)
                            condition_event[inds_event] = True
                        condition_array = condition_array | condition_event
                    # indices of samples that will be kept
                    inds_event_kept_all = np.where(condition_array==True)[0]

                    # updating the metadata, according to the subsampling
                    metadata = metadata[condition_array]
                    # EEG data subsampling is done here
                    epochs = epochs[inds_event_kept_all]

                ####################################
                epochs = epochs.get_data()
                n_trials = epochs.shape[0]
                ####################################

                # Filling the list to uniquely identify samples. This will be used in the dataloader
                for trial_index in range(n_trials):
                    list_sample_indices.append([dataset_index, trial_index])

                list_merged_ntrials.append(n_trials)
                cnt_trials += n_trials
                dataset_index += 1

                list_merged_subject.append(sub)
                list_merged_session.append(sess)
                list_merged_metadata.append(metadata)
                list_merged_epochs.append(epochs)

        self.subject = list_merged_subject
        self.session = list_merged_session
        self.ntrials = list_merged_ntrials
        self.metadata = list_merged_metadata
        self.epochs = list_merged_epochs

        ###########################

        self.metadata_concat = pd.concat(self.metadata)
        # number of sub-datasets
        self.N_datasets = len(self.subject)
        # total number of trials for the whole dataset
        self.N_trials = cnt_trials
        # final list of samples (their indices)
        self.sample_indices = list_sample_indices
        # number of EEG channels (in some cases, this does not correspond exactly to #Electrodes)
        self.N_electrodes = self.epochs[0].shape[1]
        # temporal length of EEG epochs
        self.window_size = self.epochs[0].shape[2]

        # Compute trial-wise and session-wise covariances for each sub-dataset
        self.compute_covariances()

        # Split subjects into subsets
        if self.use_subsets and self.phase=='train':
            self.create_subsets()

        self.transform = ToTensor()

    def compute_covariances(self):

        self.trialwise_covs_per_dataset = list()
        self.dataset_covs = list()
        for i in range(self.N_datasets):
            data = self.epochs[i]
            N_channels = data.shape[1]
            trial_covs = Covariances('oas').fit_transform(data)
            self.trialwise_covs_per_dataset.append(trial_covs)
            dataset_cov = mean_riemann(trial_covs, tol=1e-08, maxiter=50, init=None, sample_weight=None)
            self.dataset_covs.append(dataset_cov)
        self.trialwise_covs_per_dataset = np.array(self.trialwise_covs_per_dataset, dtype=object)
        self.dataset_covs = np.array(self.dataset_covs)
        ########################
        # Compute projection matrix inv(sqrtm(cov)) [i.e. cov^(-1/2)], that is used to align the EEG data
        projs_list = list()
        for i in range(self.N_datasets):
            dataset_cov = self.dataset_covs[i]
            proj = pinv(sqrtm(dataset_cov)).real
            projs_list.append(proj)
        projs = np.array(projs_list)
        self.projs = projs

    def covariance_align(self):

        # Align the EEG data, using the projection matrix computed based on the covariance matrices
        # This function is used to align the EEG data only once, in an offline manner,
        # i.e. before starting the training process
        for i in range(self.N_datasets):
            proj = pinv(sqrtm(self.dataset_covs[i])).real
            for i_window in range(self.ntrials[i]):
                data_epoch = self.epochs[i][i_window]
                aligned_data_epoch = np.matmul(proj, data_epoch)
                self.epochs[i][i_window] = aligned_data_epoch

    def create_subsets(self):

        self.subset_dict = dict()
        print('Randomly assigning subjects to subsets')
        subset_inds = np.random.randint(low=0, high=self.n_subsets, size=self.N_datasets)
        for i in range(self.N_datasets):
            dataset_to_subset_index = subset_inds[i]
            self.subset_dict.update({i: dataset_to_subset_index})
        unique_subset_IDs = np.sort(np.unique(subset_inds))
        if len(unique_subset_IDs)<self.n_subsets:
            print('Number of subsets ({}) is smaller than expected ({}). Quitting...'.format(len(unique_subset_IDs), self.n_subsets))
            quit()

    def get_subset(self, dataset_index):
        subset = self.subset_dict[dataset_index]
        return subset

    def __len__(self):
        return self.N_trials

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        dataset_index = self.sample_indices[idx][0]
        trial_index = self.sample_indices[idx][1]

        # Getting data of the sample
        x = self.epochs[dataset_index][trial_index].copy()
        # Getting label of the sample
        y = self.metadata[dataset_index]['target'].values[trial_index]

        subj = self.subject[dataset_index]
        sess = self.session[dataset_index]

        # Get subset index
        if self.use_subsets and self.phase=='train':
            subset = self.get_subset(dataset_index)

        # Mapping labels from all datasets, to have the same class-value correspondence
        y = self.label_correspondence[y]

        sample = {'eeg': x,
                  'target': y,
                  'dataset_idx': dataset_index,
                  'subject': subj,
                  'session': sess}
        if self.use_subsets and (self.phase=='train'):
                sample['subset'] = subset
        if self.transform:
            sample = self.transform(sample)

        return sample

def intersection(list_1, list_2):
    # find common elements from two lists
    list_3 = [value for value in list_1 if value in list_2]
    return list_3

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):

        # sample_tensor = {'eeg': torch.from_numpy(sample['eeg']),
        #                 'target': torch.from_numpy(np.array(sample['target'])).type(torch.LongTensor),
        #                  'dataset_idx': sample['dataset_idx'],
        #                  }
        # if 'subset' in sample.keys():
        #     sample_tensor.update({'subset': sample['subset']})
        # if 'subject' in sample.keys():
        #     sample_tensor.update({'subject': sample['subject']})
        # if 'session' in sample.keys():
        #     sample_tensor.update({'session': sample['session']})
        # return sample_tensor

        sample['eeg'] = torch.from_numpy(sample['eeg'])
        sample['target'] = torch.from_numpy(np.array(sample['target'])).type(torch.LongTensor)

        return sample
