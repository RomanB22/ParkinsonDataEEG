## Libraries
import os
from pathlib import Path
import matplotlib.pyplot as plt
import mne

## Setting up the environment
plt.ioff()  # Use plt.show() to show the plots

## Helper functions
def gather_all_setFile_paths() -> list: 
    """Convenience function to create a list of paths to the the EEGLab setfiles."""
    # ** indicates cwd and recursive subdirectories - https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
    return list(Path("./datos_PD_Anjum/").glob("**/*.set"))

## Gathering all the file paths
filePaths = gather_all_setFile_paths()

print(f"There are {len(filePaths)} paths.")

print("These are the first 5 paths: ")
filePaths[:5]  # Preview the top five

## Read the files
filePath = filePaths[5]  # Plot the 6th entry (list is 0-indexed)
rawEegLab = mne.io.read_raw_eeglab(filePath)  # Read the file as a raw file

print(f"Number of channels: {len(rawEegLab.ch_names)}")
print(f"Number of time points: {rawEegLab.n_times}")

## Plot the raw data
rawEegLab.plot(
    n_channels=20, 
    color="blue",
    start=5,     # Starting at 5sec
    duration=10  # Plot 10sec long --> 5 ~ 5+10 sec
)
plt.show()

## Plot the sensor names and location
rawEegLab.plot_sensors(show_names=True)
plt.show()

## Plot the Power Spectral Density (PSD) in decibel
rawEegLab.compute_psd().plot(dB=True)
plt.show()
