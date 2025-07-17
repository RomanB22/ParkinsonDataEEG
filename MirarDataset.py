from utils import *
import os
from pathlib import Path
import scipy
import matplotlib.pyplot as plt
import numpy as np
import mne
from matplotlib import ticker
from tqdm.auto import tqdm
import textwrap
from typing import Union, List
from tqdm.auto import tqdm, trange
import pandas as pd
import numpy as np
import seaborn as sns
import warnings

relativePath = './datos_PD_Anjum'

## Getting all the paths to the EEGLab '.set' files
filePaths = gather_all_setFile_paths(relativePath)

## EDA
ith_path = 5

## Extract and read the EEGLab set/fdt file
filePath = filePaths[ith_path]
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

rawEegLab.compute_psd().plot_topomap()
plt.show()



warnings.filterwarnings("ignore")

subject_info = create_df_subject_info(relativePath)
subject_info.drop("TYPE", axis=1, inplace=True)

sns.pairplot(subject_info, hue="GROUP", palette={"PD":'Red' , "Control": 'Blue'})

plt.show()

subject_info = create_df_subject_info(relativePath)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

## Plotting PD vs Control
subject_info.GROUP.value_counts().plot(kind='bar', ax=axs[0])
axs[0].bar_label(axs[0].containers[0])
axs[0].set_title("Distribution of of PD vs Control data")
axs[0].tick_params(labelrotation=0)
## Chaning the facecolor of the rectangles
for idx, rec in enumerate(axs[0].containers[0].patches): 
    colormap = ['red', 'blue']
    rec.set(facecolor=colormap[idx])
    
## Plotting Gender
subject_info.GENDER.value_counts().plot(kind='bar', ax=axs[1])
axs[1].bar_label(axs[1].containers[0])
axs[1].set_title("Distribution of Gender")
axs[1].tick_params(labelrotation=0)
## Chaning the facecolor of the rectangles
for idx, rec in enumerate(axs[1].containers[0].patches): 
    colormap = ['lightblue', 'pink']
    rec.set(facecolor=colormap[idx])

## Plotting Age
subject_info.AGE.plot(kind='hist', ax=axs[2], bins=20, color='green')
axs[2].set_title("Distribution of Age")
axs[2].set_xlabel("Age")
axs[2].set_ylabel("Count")

## Figure
fig.suptitle("Dataset profile", size='20')

plt.show()

## EDA - Understanding the data - Plotting raw data dimensions
def add_jitter(value, scale=1): 
    return value + np.random.uniform(-1, 1)*scale

## Find the entire space of channel_names
holder_ch_names = []
for idx, filepath in tqdm(enumerate(filePaths), total=len(filePaths)):
    raw = mne.io.read_raw_eeglab(filepath)
    ch_names = raw.ch_names
    holder_ch_names.extend(ch_names)
all_ch_names = list(set(holder_ch_names))

subject_info = create_df_subject_info(relativePath)

fig, ax = plt.subplots(figsize=(5, 6))
for filePath in tqdm(filePaths): 
    ## Reading the raw EEG
    raw = mne.io.read_raw_eeglab(filePath)
    n_channels = len(raw.ch_names)
    n_times = raw.n_times
    
    ## Determinig if PD or control
    participant_id = filePath.stem[:7]
    mask = subject_info.participant_id == participant_id
    label = subject_info.loc[mask, 'GROUP'].values[0]
    
    ## Plotting the scatter plot
    if label == "PD": 
        ax.scatter(add_jitter(n_channels, 0.2), n_times, marker=".", color="red", label="PD")
    elif label == "Control": 
        ax.scatter(add_jitter(n_channels, 0.2), n_times, marker=".", color="blue", label="Control")
    else: 
        print(f"Something is wrong at {filePath}")

## Add labels to the plot
ax.set_title("n_dataPoints v n_channels\n(Values jittered to avoid overlap)")
ax.set_xlabel("Number of EEG channels")
ax.set_ylabel("Number of data points per entry (EEG records at 500Hz)")
ax.set_ylim(50000, 180000)
ax.xaxis.set_major_locator(ticker.FixedLocator(range(60, 68)))
ax.grid(visible=True, which="both", alpha=0.3)

ax.text(64, 16e4, textwrap.fill("All recordings are at least 60k long, and all but one entry has 63 or 64 channels.", 30))
ax.annotate(
    "There are no 65 channels", 
    xy=(65, 6e4), 
    xytext=(65, 10e4), 
    arrowprops=dict(alpha=0.4), 
    ha="right"
)
ax.annotate(
    textwrap.fill("This may be the one with duplicate names on a single channel.", 30), 
    xy=(66, 6.3e4), 
    xytext=(65.5, 12e4),
    arrowprops=dict(alpha=0.4), 
    ha="right"
)
plt.show()