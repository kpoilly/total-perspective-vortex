import os
import sys
import mne


def visualize_data(raw):
    """
    Visualize the raw data.

    Args:
        raw (mne.io.Raw): The raw data.
    """
    try:
        raw.plot(block=True, title="Raw Data")
        raw.plot_psd(average=True, spatial_colors=False, picks='eeg',
                    title=f'PSD Données Brutes',
                    block=True
                    )
    except Exception as e:
        print(f"Error during visualization: {e}")


def main():
    if len(sys.argv) < 2:
        print("Missiong argument: python preprocessing.py <data_path>")
        sys.exit(1)

    data_path = sys.argv[1]
    if not os.path.exists(data_path):
        print(f"Data path does not exist: {data_path}")
        sys.exit(1)
    
    try:
        raw = mne.io.read_raw_edf(data_path, preload=True, verbose=False)
        print(f"{data_path} successfully loaded.")
        print(raw.info)
        visualize_data(raw)
    except Exception as e:
        print(f"Error: {e}")



if __name__ == '__main__':
    main()
