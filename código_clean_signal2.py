import os
from pathlib import Path
import seaborn as sns
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
rawEegLab = mne.io.read_raw_eeglab(filePath)  # Read the file as a raw file

print(f"Number of channels: {len(rawEegLab.ch_names)}")
print(f"Number of time points: {rawEegLab.n_times}")
print(rawEegLab.info)

rawEegLab.load_data()
rawEegLab.plot()
plt.title("Señal original")
plt.show()

rawEegLab.filter(l_freq=1, h_freq=30)  # Pasa-Banda 1-30 Hz
rawEegLab.notch_filter(50)  # Notch 50 Hz (o 60 Hz según país)

# ICA para artefactos
ica = mne.preprocessing.ICA(n_components=20)
ica.fit(rawEegLab)
ica.exclude = [0, 1]  # Ejemplo: excluir componentes de parpadeo
raw_clean = ica.apply(rawEegLab)

# Graficar señal limpia
raw_clean.plot()
plt.title("Señal limpia")
plt.show()

#raw_clean.save('eeg_limpio.fif', overwrite=True)

""" filePath = filePaths[5]
rawEegLab = mne.io.read_raw_eeglab(filePath)
#datos = rawEegLab.times
datos = rawEegLab.get_data()
#nombres = rawEegLab.ch_names
#print(nombres)
posiciones = []
datosF = [] """
""" pos = 0
for i in nombres:
    if(i[0]=='F'):
        print(i)
        print(pos)
        posiciones.append(pos)
        datosF.append(datos[pos])
    
    pos = pos + 1

print(len(posiciones))
print(len(datosF)) """

""" print(datos)
plt.boxplot(datos)
plt.show() """