def get_subject_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        subject_list = subjects_BCI_IV_2a
    elif dataset_name=='Cho2017':
        subject_list = subjects_Cho2017
    elif dataset_name=='PhysionetMI':
        subject_list = subjects_PhysionetMI
    elif dataset_name=='Lee2019_MI':
        subject_list = subjects_Lee2019_MI
    elif dataset_name=='BNCI2014004':
        subject_list = subjects_BNCI2014004

    return subject_list

def get_target_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        target_list = targets_BCI_IV_2a
    elif dataset_name=='Cho2017':
        target_list = targets_Cho2017
    elif dataset_name=='PhysionetMI':
        target_list = targets_PhysionetMI
    elif dataset_name=='Lee2019_MI':
        target_list = targets_Lee2019_MI
    elif dataset_name == 'BNCI2014004':
        target_list = targets_BNCI2014004

    return target_list

def get_event_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        event_list = events_BCI_IV_2a
    elif dataset_name=='Cho2017':
        event_list = events_Cho2017
    elif dataset_name=='PhysionetMI':
        event_list = events_PhysionetMI
    elif dataset_name == 'Lee2019_MI':
        event_list = events_Lee2019_MI
    elif dataset_name == 'BNCI2014004':
        event_list = events_BNCI2014004

    return event_list

def get_channel_list(dataset_name):

    if dataset_name=='BCI_IV_2a':
        channel_list = channels_BCI_IV_2a
    elif dataset_name=='Cho2017':
        channel_list = channels_Cho2017
    elif dataset_name=='PhysionetMI':
        channel_list = channels_PhysionetMI
    elif dataset_name == 'Lee2019_MI':
        channel_list = channels_Lee2019_MI
    elif dataset_name == 'BNCI2014004':
        channel_list = channels_BNCI2014004

    return channel_list
###############################################
# Subjects

subjects_BCI_IV_2a = [x for x in range(1, 10)]
subjects_Cho2017 = [x for x in range(1, 53) if x not in [32, 46, 49]]
subjects_PhysionetMI = [x for x in range(1, 110) if x not in [88, 90, 92, 100, 104, 106]]
subjects_Lee2019_MI = [x for x in range(1, 55)]
subjects_BNCI2014004 = [x for x in range(1, 10)]

###############################################
# Targets

# Kept targets
targets_BCI_IV_2a = ['left_hand', 'right_hand']
targets_Cho2017 = ['left_hand', 'right_hand']
targets_PhysionetMI = ['left_hand', 'right_hand']
targets_Lee2019_MI = ['left_hand', 'right_hand']
targets_BNCI2014004 = ['left_hand', 'right_hand']

# Full targets
# targets_BCI_IV_2a = ['feet', 'left_hand', 'right_hand', 'tongue']
# targets_Cho2017 = ['left_hand', 'right_hand']
# targets_PhysionetMI = ['left_hand', 'rest', 'right_hand', 'feet', 'hands']
# targets_Lee2019_MI = ['left_hand', 'right_hand']
# targets_BNCI2014004 = ['left_hand', 'right_hand']

###############################################
# Events

events_BCI_IV_2a = {'feet': 0, 'left_hand': 1, 'right_hand': 2, 'tongue': 3}
events_Cho2017 = {'left_hand': 0, 'right_hand': 1}
events_PhysionetMI = {'left_hand': 0, 'rest': 1, 'right_hand': 2, 'feet': 3, 'hands': 4}
events_Lee2019_MI = {'right_hand': 0, 'left_hand': 1}
events_BNCI2014004 = {'left_hand': 0, 'right_hand': 1}

###############################################
# Electrodes

channels_BCI_IV_2a = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1',\
                     'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

channels_Cho2017 = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',\
                    'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz',\
                    'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4',\
                    'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8',\
                    'P10', 'PO8', 'PO4', 'O2']

channels_PhysionetMI = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',\
                        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz',\
                        'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8',\
                        'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3',\
                        'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz']

# Full list of 64 channels
# channels_Lee2019_MI = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4',\
#                        'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1',\
#                        'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2',\
#                        'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',\
#                        'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']
# List of 20 channels
channels_Lee2019_MI = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'Cz', 'C4', 'CP5', 'CP1', 'CP2', 'CP6', 'FC3', 'FC4', \
                       'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4']

channels_BNCI2014004 = ['C3', 'Cz', 'C4']