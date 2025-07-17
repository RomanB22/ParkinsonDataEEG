# Verify installations
import sys
print(f"Python version: {sys.version}")

try:
    import mne
    print(f"MNE version: {mne.__version__}")
except ImportError:
    print("MNE not installed properly")

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    print("TensorFlow not installed properly")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError:
    print("NumPy not installed properly")

try:
    import sklearn
    print(f"Scikit-learn version: {sklearn.__version__}")
except ImportError:
    print("Scikit-learn not installed properly")

# Set warning level
import warnings
warnings.filterwarnings('ignore')

# Cell 1: Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne
import glob
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# Cell 2: Load and explore the dataset
# Define path to dataset
# Note: Adjust this path based on your Kaggle environment
data_path = '../input/rest-eyes-open-parkinsons-disease-64-channel-eeg/'

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

# Cell testing: Split data into train and test sets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Simulasi data (Pastikan untuk mengganti ini dengan dataset yang sebenarnya)
# Contoh dataset dengan 100 sampel dan 5 fitur
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)  # Kelas biner (0 atau 1)

# Periksa bentuk data sebelum split
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Pastikan X dan y memiliki jumlah sampel yang sama
if len(X) != len(y):
    raise ValueError("Jumlah sampel pada X dan y harus sama!")

# Split data menjadi train dan test
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
except ValueError as e:
    print(f"Errors at split data: {e}")

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Simulasi data (Pastikan untuk mengganti ini dengan dataset yang sebenarnya)
# Contoh dataset dengan 100 sampel dan 5 fitur
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, size=100)  # Kelas biner (0 atau 1)

# Periksa bentuk data sebelum split
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Pastikan X dan y memiliki jumlah sampel yang sama
assert len(X) == len(y), "Jumlah sampel pada X dan y harus sama!"

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Periksa bentuk data setelah split
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# Cell 7: Visualize EEG signals
import numpy as np
import matplotlib.pyplot as plt

# Cell 7: Visualize EEG signals
def visualize_eeg(data, title, n_channels=5, duration=2):
    plt.figure(figsize=(15, 10))
    
    # Pastikan data memiliki bentuk yang benar
    if len(data.shape) == 1:  # Jika data 1D, ubah menjadi 2D (1 channel, samples)
        data = data.reshape(1, -1)
    elif len(data.shape) == 2 and data.shape[0] < data.shape[1]:
        data = data  # Data sudah dalam bentuk (channels, samples)
    else:
        raise ValueError("Format data EEG tidak sesuai. Harus berbentuk (channels, samples).")
    
    # Select n random channels to display
    n_total_channels = data.shape[0]  # Jumlah channel yang tersedia
    n_channels = min(n_channels, n_total_channels)  # Hindari error jika jumlah channel lebih sedikit
    channels_to_plot = np.random.choice(n_total_channels, n_channels, replace=False)
    
    for i, channel in enumerate(channels_to_plot):
        plt.subplot(n_channels, 1, i+1)
        plt.plot(data[channel, :])
        plt.title(f'Channel {channel}')
        plt.ylabel('Amplitude')
        if i == n_channels - 1:
            plt.xlabel('Time (samples)')
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# Visualize a PD sample
pd_indices = np.where(y_train == 1)[0]
if len(pd_indices) > 0:
    pd_idx = pd_indices[0]
    visualize_eeg(X_train[pd_idx].T, "Parkinson's Disease EEG Signal")
else:
    print("Tidak ada sampel Parkinson's Disease dalam y_train.")

# Visualize a Control sample
control_indices = np.where(y_train == 0)[0]
if len(control_indices) > 0:
    control_idx = control_indices[0]
    visualize_eeg(X_train[control_idx].T, "Control EEG Signal")
else:
    print("Tidak ada sampel Control dalam y_train.")

# Cell 8: Visualize frequency domain
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Cell 8: Visualize Frequency Domain
def visualize_frequency_domain(data, title, sampling_rate=250, n_channels=5):
    plt.figure(figsize=(15, 10))
    
    # Pastikan data memiliki bentuk yang benar
    if len(data.shape) == 1:  # Jika data 1D, ubah menjadi 2D (1 channel, samples)
        data = data.reshape(1, -1)
    elif len(data.shape) == 2 and data.shape[0] < data.shape[1]:
        data = data  # Data sudah dalam bentuk (channels, samples)
    else:
        raise ValueError("Format data EEG tidak sesuai. Harus berbentuk (channels, samples).")
    
    # Select random channels
    n_total_channels = data.shape[0]  # Jumlah channel yang tersedia
    n_channels = min(n_channels, n_total_channels)  # Hindari error jika jumlah channel lebih sedikit
    channels_to_plot = np.random.choice(n_total_channels, n_channels, replace=False)
    
    for i, channel in enumerate(channels_to_plot):
        plt.subplot(n_channels, 1, i+1)
        
        # Calculate power spectral density
        f, Pxx = signal.welch(data[channel, :], fs=sampling_rate, nperseg=256)
        
        # Plot only frequencies up to 50 Hz
        mask = f <= 50
        plt.semilogy(f[mask], Pxx[mask])
        plt.title(f'Channel {channel}')
        plt.ylabel('PSD [V^2/Hz]')
        if i == n_channels - 1:
            plt.xlabel('Frequency [Hz]')
    
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

# Visualize frequency domain for PD
pd_indices = np.where(y_train == 1)[0]
if len(pd_indices) > 0:
    pd_idx = pd_indices[0]
    visualize_frequency_domain(X_train[pd_idx].T, "Parkinson's Disease EEG Frequency Domain")
else:
    print("Tidak ada sampel Parkinson's Disease dalam y_train.")

# Visualize frequency domain for Control
control_indices = np.where(y_train == 0)[0]
if len(control_indices) > 0:
    control_idx = control_indices[0]
    visualize_frequency_domain(X_train[control_idx].T, "Control EEG Frequency Domain")
else:
    print("Tidak ada sampel Control dalam y_train.")