

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

Number of Parkinson's Disease EEG files: 0
Number of Control EEG files: 0

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

Total number of segments: 2
Segment shape: (0,) (channels × time points)
Number of PD segments: 0
Number of Control segments: 0

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
    print(f"Error saat split data: {e}")
    print("Pastikan y memiliki variasi kelas yang cukup untuk stratify.")X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

  File "<ipython-input-10-5ada5d4c9c63>", line 26
    print("Pastikan y memiliki variasi kelas yang cukup untuk stratify.")X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                                                                         ^
SyntaxError: invalid syntax

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

# Cell 9: Prepare data for deep learning
# Reshape data for RNN: (samples, time steps, features)

# Periksa bentuk awal X_train
print(f"Original X_train shape: {X_train.shape}")

# Pastikan X_train memiliki 3 dimensi sebelum transpose
if len(X_train.shape) == 2:  # Jika data 2D (samples, features), ubah ke 3D (samples, time, channels)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Reshape data untuk RNN: (samples, time steps, features)
if X_train.shape[1] > X_train.shape[2]:  # Jika time steps tidak berada di axis-1
    X_train_rnn = X_train.transpose(0, 2, 1)
    X_test_rnn = X_test.transpose(0, 2, 1)
else:
    X_train_rnn = X_train
    X_test_rnn = X_test

print(f"RNN input shape: {X_train_rnn.shape}")

# Reshape data untuk CNN (EfficientNet): (samples, time steps, features)
X_train_cnn = X_train_rnn.copy()
X_test_cnn = X_test_rnn.copy()

print(f"CNN input shape: {X_train_cnn.shape}")

# Reshape data untuk Autoencoder: (samples, features)
X_train_ae = X_train.reshape(X_train.shape[0], -1)
X_test_ae = X_test.reshape(X_test.shape[0], -1)

print(f"Autoencoder input shape: {X_train_ae.shape}")

# Cell 10: Model 1 - Recurrent Neural Network (RNN/LSTM)
def create_rnn_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

rnn_model = create_rnn_model(X_train_rnn.shape[1:])
rnn_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

rnn_history = rnn_model.fit(
    X_train_rnn, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Cell 11: Model 2 - CNN (EfficientNet-inspired)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Dense, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Pastikan dataset memiliki dimensi yang benar
X_train_cnn = np.random.rand(1000, 32)  # Contoh data dengan (1000 sampel, 32 fitur)
y_train = np.random.randint(0, 2, 1000)  # Label biner

# Ubah bentuk data menjadi (samples, time_steps, channels)
X_train_cnn = np.expand_dims(X_train_cnn, axis=-1)  # Menambahkan dimensi channel

# Pastikan bentuk input benar
print("Shape of X_train_cnn:", X_train_cnn.shape)  # Harus (1000, 32, 1)

def create_efficientnet_inspired_model(input_shape):
    model = Sequential([
        # First block
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second block
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third block
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        
        # Global features
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Callback untuk early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Buat model
cnn_model = create_efficientnet_inspired_model(X_train_cnn.shape[1:])
cnn_model.summary()

# Training model
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Cell 12: Model 3 - Autoencoder
def create_autoencoder_model(input_dim):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(512, activation='relu')(input_layer)
    encoder = Dropout(0.3)(encoder)
    encoder = Dense(256, activation='relu')(encoder)
    encoder = Dropout(0.3)(encoder)
    encoder = Dense(128, activation='relu')(encoder)
    encoder = Dropout(0.3)(encoder)
    
    # Bottleneck layer
    bottleneck = Dense(64, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(128, activation='relu')(bottleneck)
    decoder = Dropout(0.3)(decoder)
    decoder = Dense(256, activation='relu')(decoder)
    decoder = Dropout(0.3)(decoder)
    decoder = Dense(512, activation='relu')(decoder)
    decoder = Dropout(0.3)(decoder)
    
    # Output layer
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)
    
    # Create autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Create encoder model for feature extraction
    encoder_model = Model(inputs=input_layer, outputs=bottleneck)
    
    return autoencoder, encoder_model

# Train the autoencoder
autoencoder, encoder = create_autoencoder_model(X_train_ae.shape[1])
autoencoder.summary()

ae_history = autoencoder.fit(
    X_train_ae, X_train_ae,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Extract features using the encoder
X_train_encoded = encoder.predict(X_train_ae)
X_test_encoded = encoder.predict(X_test_ae)

# Build a classifier on the encoded features
ae_classifier = Sequential([
    Dense(32, activation='relu', input_shape=(64,)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ae_classifier.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

ae_classifier_history = ae_classifier.fit(
    X_train_encoded, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Cell 13: Evaluate Model 1 - RNN
def evaluate_model(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc
    }

# Evaluate RNN model
rnn_metrics = evaluate_model(rnn_model, X_test_rnn, y_test, "RNN/LSTM Model")

# Cell 14: Evaluate Model 2 - CNN (EfficientNet)
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Dense, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping

# Pastikan dataset memiliki dimensi yang benar
X_train_cnn = np.random.rand(1000, 32)  # Contoh data dengan (1000 sampel, 32 fitur)
y_train = np.random.randint(0, 2, 1000)  # Label biner

# Ubah bentuk data menjadi (samples, time_steps, channels)
X_train_cnn = np.expand_dims(X_train_cnn, axis=-1)  # Menambahkan dimensi channel

# Pastikan bentuk input benar
print("Shape of X_train_cnn:", X_train_cnn.shape)  # Harus (1000, 32, 1)

def create_efficientnet_inspired_model(input_shape):
    model = Sequential([
        # First block
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second block
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third block
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        
        # Global features
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Callback untuk early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Buat model
cnn_model = create_efficientnet_inspired_model(X_train_cnn.shape[1:])
cnn_model.summary()

# Training model
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluasi model
def evaluate_model(model, X_test, y_test, model_name):
    # Pastikan input memiliki dimensi yang benar
    X_test = np.expand_dims(X_test, axis=-1) if X_test.shape[-1] != 1 else X_test
    print("Shape of X_test:", X_test.shape)
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = np.mean(y_pred == y_test)
    print(f"Accuracy of {model_name}: {accuracy:.4f}")
    return accuracy

# Contoh data testing
X_test_cnn = np.random.rand(20, 32)  # Contoh data test
X_test_cnn = np.expand_dims(X_test_cnn, axis=-1)  # Pastikan dimensi benar
y_test = np.random.randint(0, 2, 20)

cnn_metrics = evaluate_model(cnn_model, X_test_cnn, y_test, "CNN (EfficientNet-inspired) Model")

# Cell 15: Evaluate Model 3 - Autoencoder
ae_metrics = evaluate_model(ae_classifier, X_test_encoded, y_test, "Autoencoder Model")

# Cell 16: Compare all models
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Dense, Flatten, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Pastikan dataset memiliki dimensi yang benar
X_train_cnn = np.random.rand(1000, 32)  # Contoh data dengan (1000 sampel, 32 fitur)
y_train = np.random.randint(0, 2, 1000)  # Label biner

# Ubah bentuk data menjadi (samples, time_steps, channels)
X_train_cnn = np.expand_dims(X_train_cnn, axis=-1)  # Menambahkan dimensi channel

# Pastikan bentuk input benar
print("Shape of X_train_cnn:", X_train_cnn.shape)  # Harus (1000, 32, 1)

def create_efficientnet_inspired_model(input_shape):
    model = Sequential([
        # First block
        Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Second block
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Third block
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),
        
        # Global features
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Callback untuk early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Buat model
cnn_model = create_efficientnet_inspired_model(X_train_cnn.shape[1:])
cnn_model.summary()

# Training model
cnn_history = cnn_model.fit(
    X_train_cnn, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluasi model
def evaluate_model(model, X_test, y_test, model_name):
    # Pastikan input memiliki dimensi yang benar
    X_test = np.expand_dims(X_test, axis=-1) if X_test.shape[-1] != 1 else X_test
    print("Shape of X_test:", X_test.shape)
    
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    print(f"Accuracy of {model_name}: {accuracy:.4f}")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc
    }

# Contoh data testing
X_test_cnn = np.random.rand(20, 32)  # Contoh data test
X_test_cnn = np.expand_dims(X_test_cnn, axis=-1)  # Pastikan dimensi benar
y_test = np.random.randint(0, 2, 20)

cnn_metrics = evaluate_model(cnn_model, X_test_cnn, y_test, "CNN (EfficientNet-inspired) Model")

# Plot perbandingan
models = ["CNN (EfficientNet)"]
metrics = [cnn_metrics]

plt.figure(figsize=(10, 6))

# Accuracy
plt.subplot(2, 2, 1)
plt.bar(models, [m['accuracy'] for m in metrics])
plt.title('Accuracy Comparison')
plt.ylim(0, 1)

# Precision
plt.subplot(2, 2, 2)
plt.bar(models, [m['precision'] for m in metrics])
plt.title('Precision Comparison')
plt.ylim(0, 1)

# Recall
plt.subplot(2, 2, 3)
plt.bar(models, [m['recall'] for m in metrics])
plt.title('Recall Comparison')
plt.ylim(0, 1)

# F1-score
plt.subplot(2, 2, 4)
plt.bar(models, [m['f1'] for m in metrics])
plt.title('F1-Score Comparison')
plt.ylim(0, 1)

plt.tight_layout()
plt.suptitle('Model Performance Comparison', fontsize=16)
plt.subplots_adjust(top=0.9)
plt.show()

# AUC comparison
plt.figure(figsize=(10, 6))
plt.bar(models, [m['auc'] for m in metrics])
plt.title('AUC Comparison')
plt.ylim(0, 1)
plt.show()

# Cell 17: Summary and Conclusion
import numpy as np

# Summary of Model Performance
print("Summary of Model Performance:")
print("-" * 70)

for i, model_name in enumerate(models):
    print(f"{model_name}:")
    print(f"  Accuracy: {metrics[i]['accuracy']:.4f}")
    print(f"  Precision: {metrics[i]['precision']:.4f}")
    print(f"  Recall: {metrics[i]['recall']:.4f}")
    print(f"  F1-score: {metrics[i]['f1']:.4f}")
    print(f"  AUC: {metrics[i]['auc']:.4f}")
    print("-" * 70)

# Determine the best models based on key metrics
best_acc_idx = np.argmax([m['accuracy'] for m in metrics])
best_f1_idx = np.argmax([m['f1'] for m in metrics])
best_auc_idx = np.argmax([m['auc'] for m in metrics])

print("Best Models Based on Performance Metrics:")
print(f"  Best model by Accuracy: {models[best_acc_idx]}")
print(f"  Best model by F1-score: {models[best_f1_idx]}")
print(f"  Best model by AUC: {models[best_auc_idx]}")
print("\nConclusions:")

# Model-wise analysis and conclusions
print("1. Recurrent Neural Networks (RNN/LSTM):")
print("   - This model captures temporal dependencies well, making it effective for sequential EEG signal analysis.")
print("   - Suitable for time-series problems, but may require more computational power due to sequential processing.")
print("   - May suffer from vanishing gradients, requiring careful tuning of hyperparameters and architectures like bidirectional LSTMs.")
print("\n2. CNN (EfficientNet-inspired Model):")
print("   - CNNs are excellent at capturing spatial patterns in EEG signals.")
print("   - EfficientNet-inspired architecture introduces depth and batch normalization for stability.")
print("   - Works well with well-preprocessed data but may not capture temporal dependencies as effectively as RNNs.")
print("\n3. Autoencoders:")
print("   - Used for feature extraction and dimensionality reduction.")
print("   - May not perform as well on classification tasks alone, but when combined with another classifier, it enhances feature learning.")
print("   - Particularly useful for anomaly detection tasks and unsupervised learning.")

print("Overall, the best-performing model depends on the evaluation metric used.")
print(f"For classification tasks with a focus on balance between precision and recall, {models[best_f1_idx]} performs the best.")
print(f"For distinguishing between classes effectively, {models[best_auc_idx]} is the best choice.")