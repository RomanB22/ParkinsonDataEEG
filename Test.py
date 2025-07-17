# Cell 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne
import glob
from scipy import signal
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Cell 2: Load and explore the dataset
# Define path to dataset
# Note: Adjust this path based on your Kaggle environment
data_path = './datos_separados_parkinson_control'

# Get list of files
pd_files = glob.glob(os.path.join(data_path, 'PD', '*.edf'))
control_files = glob.glob(os.path.join(data_path, 'Control', '*.edf'))

print(f"Number of Parkinson's Disease EEG files: {len(pd_files)}")
print(f"Number of Control EEG files: {len(control_files)}")

# Load a sample file to explore
sample_pd_file = pd_files[0] if pd_files else None
sample_control_file = control_files[0] if control_files else None

if sample_pd_file:
    raw_pd = mne.io.read_raw_edf(sample_pd_file, preload=True)
    print("\nParkinson's Disease EEG Sample Info:")
    print(f"Channels: {raw_pd.info['nchan']}")
    print(f"Channel names: {raw_pd.info['ch_names'][:5]}... (showing first 5)")
    print(f"Sampling frequency: {raw_pd.info['sfreq']} Hz")
    print(f"Duration: {raw_pd.times[-1]} seconds")

if sample_control_file:
    raw_control = mne.io.read_raw_edf(sample_control_file, preload=True)
    print("\nControl EEG Sample Info:")
    print(f"Channels: {raw_control.info['nchan']}")
    print(f"Channel names: {raw_control.info['ch_names'][:5]}... (showing first 5)")
    print(f"Sampling frequency: {raw_control.info['sfreq']} Hz")
    print(f"Duration: {raw_control.times[-1]} seconds")

# Cell 3: Function to preprocess EEG data
def preprocess_eeg(raw_eeg):
    # Apply bandpass filter (0.5-45 Hz)
    raw_eeg.filter(0.5, 45., fir_design='firwin')
    
    # Get the data
    data = raw_eeg.get_data()
    
    # Normalize the data
    scaler = StandardScaler()
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        normalized_data[i, :] = scaler.fit_transform(data[i, :].reshape(-1, 1)).flatten()
    
    return normalized_data

# Cell 4: Function to extract features
def extract_features(raw_eeg, segment_duration=2.0, overlap=0.5):
    data = preprocess_eeg(raw_eeg)
    sfreq = raw_eeg.info['sfreq']
    n_channels = data.shape[0]
    
    # Calculate segment length and step in samples
    segment_length = int(segment_duration * sfreq)
    step = int((1 - overlap) * segment_length)
    
    # Calculate number of segments
    n_segments = (data.shape[1] - segment_length) // step + 1
    
    # Initialize feature array
    segments = np.zeros((n_segments, n_channels, segment_length))
    
    # Extract segments
    for i in range(n_segments):
        start = i * step
        end = start + segment_length
        segments[i, :, :] = data[:, start:end]
    
    return segments

# Cell 5: Load and process all data
def load_all_data(pd_files, control_files, max_files=None, segment_duration=2.0, overlap=0.5):
    if max_files:
        pd_files = pd_files[:max_files]
        control_files = control_files[:max_files]
    
    pd_segments_list = []
    control_segments_list = []
    
    # Process PD files
    for file in pd_files:
        try:
            raw = mne.io.read_raw_edf(file, preload=True)
            segments = extract_features(raw, segment_duration, overlap)
            pd_segments_list.append(segments)
            print(f"Processed PD file: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Process Control files
    for file in control_files:
        try:
            raw = mne.io.read_raw_edf(file, preload=True)
            segments = extract_features(raw, segment_duration, overlap)
            control_segments_list.append(segments)
            print(f"Processed Control file: {os.path.basename(file)}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    # Combine segments from all files
    pd_segments = np.vstack(pd_segments_list) if pd_segments_list else np.array([])
    control_segments = np.vstack(control_segments_list) if control_segments_list else np.array([])
    
    # Create labels (1 for PD, 0 for Control)
    pd_labels = np.ones(pd_segments.shape[0])
    control_labels = np.zeros(control_segments.shape[0])
    
    # Combine data and labels
    X = np.vstack([pd_segments, control_segments])
    y = np.hstack([pd_labels, control_labels])
    
    return X, y

# Due to computational constraints, we'll limit the number of files to process
X, y = load_all_data(pd_files, control_files, max_files=5)  # Adjust max_files based on your resources

print(f"\nTotal number of segments: {X.shape[0]}")
print(f"Segment shape: {X.shape[1:]} (channels × time points)")
print(f"Number of PD segments: {np.sum(y == 1)}")
print(f"Number of Control segments: {np.sum(y == 0)}")
