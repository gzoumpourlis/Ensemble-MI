import os
import sys
sys.dont_write_bytecode = True
import mne
mne.set_log_level('CRITICAL')
import pickle
import datetime

from braindecode.datasets import MOABBDataset
from braindecode.preprocessing.windowers import create_windows_from_events
from braindecode.preprocessing.preprocess import (preprocess, Preprocessor)

from src.config import *
from src.args_utils import create_exp_args

###################################################

# Create the experiment's name string, containing date and time
now = datetime.datetime.now()
exp_name = now.strftime("%Y_%m_%d-%H%M%S")

exp_args_dict = {
    'exp_group': exp_name,
    'dataset_name': 'PhysionetMI',
    # Dataset preprocessing
    'fmin': 4., # bandpass filter fmin (in Hz)
    'fmax': 38., # bandpass filter fmax (in Hz)
    'sfreq': 100., # resampling frequency (in Hz)
    't_start_offset': 0.0, # trial start offset (in seconds)
    't_end_offset': 0.0, # trial end offset (in seconds)
    't_dur': 4.0, # trial duration (in seconds)
    'use_car': True, # apply Common Average Reference (CAR) on EEG signals
}

def main(exp_args_dict):
    print('Preprocessing dataset: {}\n'.format(exp_args_dict['dataset_name']))
    args = create_exp_args(exp_args_dict=exp_args_dict)
    preprocess_dataset(args=args)
    print('\nFinished!')

def preprocess_dataset(args):

    dataset_name = args.dataset_name
    # Using dataset details from src/config.py
    subject_range = get_subject_list(dataset_name)
    dataset_targets = get_target_list(dataset_name)
    dataset_channels = get_channel_list(dataset_name)

    ############################
    # Preprocessing

    low_cut_hz = args.fmin  # low cut frequency for filtering
    high_cut_hz = args.fmax  # high cut frequency for filtering
    preprocessors = [
        Preprocessor(lambda x: x * 1e6),  # Convert from V to uV
        Preprocessor('pick_channels', ch_names=dataset_channels, ordered=True),  # pick electrodes, re-order
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
    ]
    # Apply Common Average Reference
    if args.use_car:
        preprocessors.append(Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'))
    # Resample
    preprocessors.append(Preprocessor('resample', sfreq=args.sfreq))

    ############################

    for subj in subject_range:
        # Get single-subject dataset
        dataset = MOABBDataset(dataset_name=dataset_name, subject_ids=[subj, ])
        # Preprocess EEG data
        preprocess(dataset, preprocessors)
        ############################
        # Apply windowing

        # Window specs
        trial_start_offset_seconds = args.t_start_offset
        trial_stop_offset_seconds = args.t_end_offset
        sfreq = dataset.datasets[0].raw.info['sfreq']
        assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
        trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
        trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)
        window_size_samples = int(args.t_dur * sfreq)

        windows_dataset = create_windows_from_events(
            dataset,
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=trial_stop_offset_samples,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples,
            drop_last_window=True,
            preload=True,
        )

        ############################

        # Save preprocessed data as .pkl file
        dataset_root_path = os.path.join(os.getcwd(),
                                        'preprocessed_data',
                                        dataset_name)
        if not os.path.exists(dataset_root_path):
            os.makedirs(dataset_root_path, exist_ok=True)
        dataset_filename = 'windows_dataset_{:03}.pkl'.format(subj)
        dataset_filepath = os.path.join(dataset_root_path, dataset_filename)
        with open(dataset_filepath, 'wb') as outp:
            pickle.dump(windows_dataset, outp, pickle.HIGHEST_PROTOCOL)
        print('Dataset: {} | Subject: #{:03d} | Saved .pkl file: {}'.format(dataset_name, subj, dataset_filepath))

if __name__ == '__main__':
    main(exp_args_dict)