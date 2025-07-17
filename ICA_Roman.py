import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA, corrmap, create_eog_epochs
from utils import *

## Setting up the environment
plt.ioff()  # Use plt.show() to show the plots

relativePath = './datos_PD_Anjum'

## Getting all the paths to the EEGLab '.set' files
filePaths = gather_all_setFile_paths(relativePath)

## Change subject
ith_path = 5

## Extract and read the EEGLab set/fdt file
filePath = filePaths[ith_path]
raw = mne.io.read_raw_eeglab(filePath)  # Read the file as a raw file

badChannels = mne.pick_channels_regexp(raw.ch_names, regexp='Fp*|AF*|FT*')
# Podes probar cambiar badChannels por solo un canal, u otros, e ir viendo como cambian todos los resultadosß
raw.info["bads"] = [raw.ch_names[i] for i in badChannels]  # Mark the bad channels
raw.plot(order=badChannels, n_channels=len(badChannels))
plt.show()

# eog_channels = ["Fp1"] #["Fp1", "Fp2"] Here we propose the channel/s with the largest artifacts as a reference. Could change
eog_channels = [raw.ch_names[i] for i in badChannels] #  Here we propose the channel/s with the largest artifacts as a reference. Could change


raw.load_data()
raw.plot()
plt.title("Señal original")
plt.show()

eog_evoked = create_eog_epochs(raw, ch_name=eog_channels).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

filt_raw = raw.copy().filter(l_freq=2.0, h_freq=None)
filt_raw.notch_filter(60)  # Notch 50 Hz (o 60 Hz según país)
ica = ICA(n_components=15, max_iter="auto", random_state=97)
ica.fit(filt_raw)
ica

raw.load_data()
ica.plot_sources(raw, show_scrollbars=True)

ica.plot_components()

# blinks
blinksComponent = [1]
ica.plot_overlay(raw, exclude=blinksComponent, picks="eeg")

suspiciousComponents = [0, 1, 2, 3, 4]
ica.plot_properties(raw, picks=suspiciousComponents)

reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(show_scrollbars=True)
reconst_raw.plot(show_scrollbars=True)
del reconst_raw

# Automatic detection of EOG artifactsica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)
ica.exclude = eog_indices

# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)

# plot diagnostics
ica.plot_properties(raw, picks=eog_indices)

# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(raw, show_scrollbars=True)

# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)

reconst_raw = raw.copy()
ica.apply(reconst_raw)

raw.plot(show_scrollbars=True)
reconst_raw.plot(show_scrollbars=True)
plt.show()
del reconst_raw