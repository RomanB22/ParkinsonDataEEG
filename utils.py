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

################################################################################
## Update MNE-Python if version is less than 1.7.*
################################################################################
# if mne.__version__[:3] != "1.7": 
#     !pip install --quiet --force-reinstall -v "mne>=1.7.*"
#     os._exit(00)

################################################################################
## Notebook level variables and settings
################################################################################
notebook_journal = [ ] # This holds the notes for the models
mne.set_log_level("ERROR")
plt.ioff()

################################################################################
## Various helper functions
################################################################################
def gather_all_setFile_paths(relativePath): 
    """Convenience function to create a list of paths to the the EEGLab setfiles."""
    # ** indicates cwd and recursive subdirectories - https://docs.python.org/3/library/pathlib.html#pathlib.Path.glob
    return list(Path(relativePath).glob("**/*.set"))


def find_set_of_common_channels(filePaths): 
    """Iterate through all the filepaths to find the set of common channel names."""
    ## Iterates through the files to find the common set of channel names
    for idx, filepath in tqdm(enumerate(filePaths), total=len(filePaths), desc="Finding the set of common channels"):
        raw = mne.io.read_raw_eeglab(filepath)
        ch_names = raw.ch_names

        if idx == 0: 
            ch_names_holder = ch_names
        else: 
            ch_names_holder = list(set(ch_names_holder) & set(ch_names))

    return ch_names_holder


def get_subject_info(subject_id:str, relativePath:str) -> pd.DataFrame: 
    """Given the subject ID, return the subject info as a dataframe.
    
    The subject info includes participant_id, GROUP, ID, EEG (EEG file name), AGE, 
    GENDER, MOCA (MOCA score), UPDRS (UPDRS score), and TYPE (numeric value of 
    patient type).
    """
    assert isinstance(subject_id, str), f"The subject_id has to be a string, got {type(subject_id)}"
    
    # Read the subject info
    path_sub_info = Path(
        relativePath,
        "participants.tsv")
    df_sub_info = pd.read_csv(path_sub_info, sep='\t')
    df_sub_info = df_sub_info[df_sub_info.participant_id == str(subject_id)]
    
    return df_sub_info


def get_subject_data(subject_id:str, relativePath:str) -> pd.DataFrame: 
    """Given the subject ID, return the raw-eeg data as a dataframe.
    
    It may be easier to work with a ndarray. However dataframe provides
    additional information such as column names that makes understanding the 
    data easier.
    """
    
    assert isinstance(subject_id, str), f"The subject_id has to be a string, got {type(subject_id)}"
    
    # Find the EEGLab set file
    path_sub_dir = Path(
        relativePath,
        str(subject_id)
    )
    path_sub_set_file = list(path_sub_dir.glob("**/*.set"))  # Generator to list
    if len(path_sub_set_file) > 1: 
        raise Exception("More than one set file is found.")
    else: 
        path_sub_set_file = path_sub_set_file[0]
        raw = mne.io.read_raw_eeglab(path_sub_set_file)
        df_raw_eeg = pd.DataFrame(raw.get_data())
        df_raw_eeg["channel_name"] = raw.ch_names
        df_raw_eeg = df_raw_eeg.set_index("channel_name")
        
        return df_raw_eeg
    

def calculate_band_power(
        filepath:Path, 
        channels:Union[str, List[str]], 
        brain_wave_bands:dict=None) -> pd.DataFrame:
    """Calculate the power of each brainwave band and return an ndarray.
    
    The bands were established by (Saboo, 2019) into 7-groups. 
    Becaue the PSD is a distrbituion, to get the power of each band, we add up
    all the PSD values of a specified range to get the power. 
    The power is calculated in V^2 and the unit of the PSD was in V^2 / Hz.
    """

    # Saboo, K. V., Varatharajah, Y., Berry, B. M., Kremen, V., Sperling, M. R., Davis, K. A., Jobst, B. C., 
    # Gross, R. E., Lega, B., Sheth, S. A., Worrell, G. A., Iyer, R. K., & Kucewicz, M. T. (2019). 
    # Unsupervised machine-learning classification of electrophysiologically active electrodes during 
    # human cognitive task performance. Scientific Reports, 9(1), Article 1. 
    # https://doi.org/10.1038/s41598-019-53925-5
    if brain_wave_bands is None: 
        brain_wave_bands = dict(
            low_theta=(2, 5), # Essentially the delta band
            high_theta=(6, 9), 
            alpha=(10, 15), 
            beta=(16, 25), 
            low_gamma=(36, 55), 
            high_gamma_1=(65, 89), 
            high_gamma_2=(90, 115),
        )

    ## Read the file, calculate PSD, extract the PSD (V^2/Hz)
    raw = mne.io.read_raw_eeglab(filepath)
    spectrum = raw.compute_psd(picks=channels, n_jobs=-1)  
    data, freqs = spectrum.get_data(return_freqs=True)
    
    holder_stats = np.ndarray((data.shape[0], len(brain_wave_bands)))
    
    ## Calculate the sum for each brainwave band across all each channels
    ## NOTE: The spectrum is in PSD (voltage^2), thus the band power is the sum of the values in the range
    for idx, (key, val) in enumerate(brain_wave_bands.items()): 
        low_bound, high_bound = val[0], val[1]
        indices = np.argwhere(
            np.logical_and(
                freqs >= low_bound, 
                freqs <= high_bound,
            )
        )
        ## Calculate the band mean across each channel
        holder_stats[:, idx] = np.sum(data[:, indices], axis=1).ravel()
        

    assert holder_stats.shape == (data.shape[0], len(brain_wave_bands)), "Something is wrong."
    
    df_stats = pd.DataFrame(holder_stats)
    df_stats.columns = brain_wave_bands.keys()
    df_stats["channel_name"] = channels
    df_stats = df_stats.set_index("channel_name")
    
    return df_stats

def create_df_subject_info(relativePath) -> pd.DataFrame: 
    # Read the subject info
    path_sub_info = Path(
        relativePath, 
        "participants.tsv")
    df_sub_info = pd.read_csv(path_sub_info, sep='\t')
    
    return df_sub_info


def create_df_psd(filePaths:List[Path], channels:List[str]=None) -> pd.DataFrame:
    if channels is None: 
        common_channels = find_set_of_common_channels(filePaths)
    
    holder_df = [ ] 
    
    for filepath in tqdm(filePaths, total=len(filePaths), desc="Creating a dataframe of PSD"): 
        raw = mne.io.read_raw_eeglab(filepath)
        spectrum = raw.compute_psd(picks=common_channels, n_jobs=-1)  
        data, freqs = spectrum.get_data(return_freqs=True)
        participant_id = filepath.stem[:7]
        df_psd = pd.DataFrame(data)
        df_psd["channel_name"] = common_channels
        df_psd["participant_id"] = [participant_id] * data.shape[0]
        df_psd = df_psd.set_index(["participant_id", "channel_name"])
        holder_df.append(df_psd)
        
    df_psd = pd.concat(holder_df)
    return df_psd

def create_df_band_power(filePaths:List[Path], channels:List[str]=None) -> pd.DataFrame: 
    if channels is None: 
        common_channels = find_set_of_common_channels(filePaths)
    
    holder_df = [ ] 
    
    for filepath in tqdm(filePaths, total=len(filePaths), desc="Creating a dataframe of band powers"): 
        df_band_power = calculate_band_power(filepath, common_channels)
        participant_id = filepath.stem[:7]
        df_band_power["participant_id"] = [participant_id] * df_band_power.shape[0]
        df_band_power = df_band_power.reset_index()  # Else the 'channel_name' is not found in df.columns
        df_band_power = df_band_power.set_index(["participant_id", "channel_name"])
        holder_df.append(df_band_power)
        
    df_band_powers = pd.concat(holder_df)
    
    return df_band_powers