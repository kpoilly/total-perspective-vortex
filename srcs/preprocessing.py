import os
import sys
import mne
import glob
import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt


def set_epochs(objs, tmin, tmax, baseline_correction):
    """
    Set epochs for the dataset
    
    Args:
        objs (dict): Dict with mne.io.Raw objs
        tmin (float): 
    """
    
    print("--- Starting Epoching Process ---")
    epochs_data = []
    epochs_labels = []
    subj_run_info = []
    runs_left_right = ['R03', 'R04', 'R07', 'R08', 'R11', 'R12']
    runs_fists_feet = ['R05', 'R06', 'R09', 'R10', 'R13', 'R14']
    event_labels = {
        'left_hand': 1,
        'right_hand': 2,
        'fists': 3,
        'feet': 4
    }
    
    for subj_id, runs in objs.items():
        for run_id, obj in runs.items():
            if obj is None:
                continue
            
            event_map = {}
            if run_id in runs_left_right:
                run_type = 'left_right_hands'
                event_map = {'T1': event_labels['left_hand'],
                             'T2': event_labels['right_hand']}
            elif run_id in runs_fists_feet:
                run_type = 'fists_feets'
                event_map = {'T1': event_labels['fists'],
                             'T2': event_labels['feet']}
            else:
                continue
            
            try:
                events_array, event_id = mne.events_from_annotations(
                    obj,
                    event_id=event_map,
                    verbose='WARNING'
                )
                if not event_id or events_array.shape[0] == 0:
                    continue
                
                epochs = mne.Epochs(obj,
                                    events=events_array,
                                    event_id=event_id,
                                    tmin=tmin,
                                    tmax=tmax,
                                    picks='eeg',
                                    baseline=baseline_correction,
                                    preload=True,
                                    event_repeated='drop',
                                    verbose='WARNING')
                if len(epochs) == 0:
                    continue
                # print(f"  Created {len(epochs)} epochs for {subj_id} - {run_id} with labels: {np.unique(epochs.events[:, -1])}")
                
                epochs_data.append(epochs.get_data(copy=False))
                epochs_labels.append(epochs.events[:, -1])
                
                for i in range(len(epochs)):
                    subj_run_info.append({
                        'subject': subj_id,
                        'run': run_id,
                        'run_type': run_type,
                        'label': epochs.events[i, -1]
                    })
            except ValueError as ve:
                print(f"  ValueError creating epochs for {subj_id} - {run_id}: {ve}")
            except Exception as e:
                print(f"  Unexpected error for {subj_id} - {run_id}: {e}")
            
    X = np.concatenate(epochs_data, axis=0)
    y = np.concatenate(epochs_labels, axis=0)

    print(f"--- Epoching Process Complete ---")
    print(f"Shape of X (all epochs data): {X.shape}")
    print(f"Shape of y (all epochs labels): {y.shape}")
    
    return X, y, subj_run_info
                               
def filter_data(raw, l_freq=1.0, h_freq=40.0, notch_freqs=None):
    """
    Filter the raw data.

    Args:
        raw (mne.io.Raw): The raw data.
        l_freq (float): The lower frequency.
        h_freq (float): The higher frequency.
        notch_freqs (float): The notch frequency.

    Returns:
        mne.io.Raw: The filtered data.
    """

    # print(f"Filtering data: band-pass [{l_freq}-{h_freq} Hz]")
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', skip_by_annotation='edge', verbose='WARNING')
    if notch_freqs:
        # print(f"Applying notch filter at {notch_freqs} Hz")
        raw_filtered.notch_filter(freqs=notch_freqs, fir_design='firwin', verbose='WARNING')
        
    return raw_filtered

def visualize_data(raw):
    """
    Visualize the raw data.

    Args:
        raw (mne.io.Raw): The raw data.
    """

    try:
        raw.plot(block=True, title="Raw Data")
        fig_psd_object = raw.compute_psd(picks='eeg', fmin=0.1, fmax=30.0).plot( # fmax en float
            average=True,
            spatial_colors=False,
            show=True
        )
        plt.show(block=True)
    except Exception as e: # TODO: Fix the PSD plot title
        print(f"Error during visualization: {e}") # TODO: Fix the PSD plot title and block
        
def load_data(data_path):
    """
    Load the data from the given path.

    Args:
        data_path (str): The path to the data.

    Returns:
        raw_objs (dict): The raw data.
        events_data (dict): The events data.
    """  

    raw_objs = {}
    events_data = {}
    subj_folders = sorted([d for d in os.listdir(data_path) \
        if os.path.isdir(os.path.join(data_path, d)) and d.startswith('S')])
    
    for subj in subj_folders:
        subj_id = subj
        subj_path = os.path.join(data_path, subj)
        
        raw_objs[subj_id] = {}
        events_data[subj_id] = {}

        edf_files = sorted(glob.glob(os.path.join(subj_path, f"{subj_id}R*.edf")))
        for edf_file in edf_files:
            run_id = os.path.basename(edf_file).replace(subj_id, '').split('.')[0]
            event_file = edf_file + ".event"
            
            print(f"--- {subj_id} - {run_id} ---")
            print(f"EDF: {edf_file}")
            print(f"Event: {event_file}")
            
            try:
                raw = mne.io.read_raw_edf(edf_file, preload=True, verbose='WARNING')
                raw_objs[subj_id][run_id] = raw
                events_data[subj_id][run_id] = raw.annotations
                print(f"  Loaded: {raw.info['nchan']} canals, {raw.n_times} samples, {raw.info['sfreq']} Hz")
                
            except Exception as e:
                print(f"Error with {edf_file}: {e}")
                raw_objs[subj_id][run_id] = None
                events_data[subj_id][run_id] = None
                
    return raw_objs, events_data


def main():
    if len(sys.argv) < 2:
        print("Missiong argument: python preprocessing.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Data path does not exist: {data_path}")
        sys.exit(1)
        
    raw_objs, events_data = load_data(data_path)
    
    # example_subj = 'S001'
    # example_run = 'R01'
    # print(f"--- {example_subj} - {example_run} ---")
    # print(f"Events:")
    # annots = events_data[example_subj][example_run]
    # for i in range(min(5, len(annots))):
    #     print(f"  Onset: {annots.onset[i]:.2f}s, Duration: {annots.duration[i]:.2f}s, Description: '{annots.description[i]}'")
    # visualize_data(raw_objs[example_subj][example_run])
    # filtered_data = filter_data(raw_objs[example_subj][example_run])
    # visualize_data(filtered_data)
    
    print("--- Starting Filtering of the Entire Dataset ---")
    filtered_objs = {}
    for subj_id, subj_data in raw_objs.items():
        filtered_objs[subj_id] = {}
        for run_id, raw in subj_data.items():
            if raw is not None:
                filtered_objs[subj_id][run_id] = filter_data(raw, l_freq=1.0, h_freq=40.0, notch_freqs=[50.0])
            else:
                filtered_objs[subj_id][run_id] = None
    print("--- Filtering of the Entire Dataset Complete ---")
    
    X, y, info_epochs = set_epochs(
    filtered_objs,
    tmin=-0.5,
    tmax=4.0,
    baseline_correction=(-0.2, 0)
    )
    
    print("--- Selecting data ---")
    event_labels = {
        'left_hand': 1,
        'right_hand': 2,
        'fists': 3,
        'feet': 4
    }
    events_tracked = ['left_hand', 'right_hand'] # A recup en argv
    print(f"Movement A = {events_tracked[0]} | Movement B = {events_tracked[1]}")
    selection = np.where(
            (y == event_labels[events_tracked[0]]) | 
            (y == event_labels[events_tracked[1]])
        )[0]
    X_select = X[selection]
    y_select =y[selection]
    print(f"Shape of X (all epochs data): {X_select.shape}")
    print(f"Shape of y (all epochs labels): {y_select.shape}")
    
    y_binary = np.zeros_like(y_select)
    y_binary[y_select == 1] = 0
    y_binary[y_select == 2] = 1
    
    print("Saving data...")
    np.savez('data/processed/data.npz',
             X_data=X_select, y_labels=y_binary)
    print("Data saved to data/processed/")

    

if __name__ == '__main__':
    main()
