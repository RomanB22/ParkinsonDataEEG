import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs

## Setting up the environment
plt.ioff()  # Use plt.show() to show the plots

## Helper functions
def gather_all_setFile_paths() -> list: 
    """Convenience function to create a list of paths to the the EEGLab setfiles."""
    # ** indicates cwd and recursive subdirectories - https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
    return list(Path("kaggle").glob("**/*.set"))
## Gathering all the file paths
filePaths = gather_all_setFile_paths()

print(f"There are {len(filePaths)} paths.")

print("These are the first 5 paths: ")
print(filePaths[:5])  # Preview the top five

filePath = filePaths[5]  # Plot the 6th entry (list is 0-indexed)
raw = mne.io.read_raw_eeglab(filePath)  # Read the file as a raw file

print(f"Number of channels: {len(raw.ch_names)}")
print(f"Number of time points: {raw.n_times}")

print(raw.info)
raw.plot()
plt.show()
raw.load_data()
# Filtrado de banda (p. ej., 1-40 Hz para EEG estándar)
raw.filter(1., 40., fir_design='firwin')

# Filtrado notch para eliminar interferencia de línea eléctrica (50 Hz o 60 Hz)
# raw.notch_filter(np.arange(50, 251, 50))  # Ajusta según tu frecuencia de red

# Crear epochs alrededor de eventos de parpadeo
eog_events = mne.preprocessing.find_eog_events(raw)
eog_epochs = mne.Epochs(raw, eog_events, tmin=-0.5, tmax=0.5, baseline=(-0.5, -0.3))

# Aplicar ICA para corrección de artefactos oculares
ica = mne.preprocessing.ICA(n_components=15, random_state=97)
ica.fit(raw)
ica.plot_components()

# Identificar componentes EOG automáticamente o manualmente
eog_indices, eog_scores = ica.find_bads_eog(raw)
ica.exclude = eog_indices

# Aplicar ICA a los datos
ica.apply(raw)

# Filtrar en alta frecuencia para resaltar artefactos musculares
raw_highpass = raw.copy().filter(l_freq=20., h_freq=None)
muscle_epochs = mne.preprocessing.create_epochs(raw_highpass, duration=1)

# Identificar componentes musculares con ICA
muscle_indices, muscle_scores = ICA.find_bads_muscle(raw_highpass)
ICA.exclude.extend(muscle_indices)