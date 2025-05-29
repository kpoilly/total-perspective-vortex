import os
import sys
import mne
import glob
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


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

    print(f"Filtering data: band-pass [{l_freq}-{h_freq} Hz]")
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin', skip_by_annotation='edge', verbose='WARNING')
    if notch_freqs:
        print(f"Applying notch filter at {notch_freqs} Hz")
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
    
    example_subj = 'S001'
    example_run = 'R01'
    
    print(f"--- {example_subj} - {example_run} ---")
    print(f"Events:")
    annots = events_data[example_subj][example_run]
    for i in range(min(5, len(annots))):
        print(f"  Onset: {annots.onset[i]:.2f}s, Duration: {annots.duration[i]:.2f}s, Description: '{annots.description[i]}'")
    visualize_data(raw_objs[example_subj][example_run])
    filtered_data = filter_data(raw_objs[example_subj][example_run])
    visualize_data(filtered_data)

if __name__ == '__main__':
    main()
