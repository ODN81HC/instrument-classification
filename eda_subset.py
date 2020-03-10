#import tarfile
import os
import librosa as lb
import numpy as np
#import json
import pandas as pd
import matplotlib.pyplot as plt
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile as wf
from tqdm import tqdm

def calc_fft(y, sr):
    n = len(y)
    freq = np.fft.rfftfreq(n, d= 1/sr)
    Y = np.abs(np.fft.rfft(y)/n)
    return (Y, freq)

def envelope(y, sr, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(sr/10), min_periods = 1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask

def graph_plot(signal, title: str, show_img: bool = True):
    fig, ax = plt.subplots(4,3, constrained_layout=True)
    fig.suptitle(title, size = 16)
    ax[-1,-1].axis('off') # Turn off the last extra plot
    for a, (key, value) in zip(ax.flatten(), signal.items()):
        a.set_title(key)
        if show_img:
            a.plot(value)
        else:
            a.imshow(value, cmap = 'hot', interpolation='nearest')
    plt.show()

def main():

    # File paths define
    folder_train = './wavfile_valid/'
    clean_train = './clean_valid/'
    # Create a list of filenames
    filenames = os.listdir(folder_train)
    # Class list
    classes = list(np.unique([filename.split('_')[0] for filename in filenames]))
    # Create a dataframe for the subset
    audio_list = []
    for filename in filenames:
        value_dict = {'fname':filename, 'instrument':filename.split('_')[0],
                      'instrument_idx':classes.index(filename.split('_')[0])}
        audio_list.append(value_dict)
    train_df = pd.DataFrame(audio_list)

    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    for c in classes:
        # Do something
        wavfile = train_df[train_df.instrument == c].iloc[0,0]
        signal, sr = lb.core.load(folder_train + wavfile, sr = None)
        mask = envelope(signal, sr, 0.0005)
        signal = signal[mask]

        signals[c] = signal
        fft[c] = calc_fft(signal, sr)

        bank = logfbank(signal[:sr], sr, nfilt = 26, nfft=1103).T
        fbank[c] = bank

        mel = mfcc(signal[:sr], sr, numcep=13, nfilt=26).T
        mfccs[c] = mel
    
    # Plot the graphs
    # graph_plot(signals, 'Instrument Audio Signal')

    # graph_plot(fft, 'Instrument FFT')

    graph_plot(fbank, 'Mel filter coefficient', show_img=False)

    if len(os.listdir(clean_train)) == 0:
        for f in tqdm(train_df.fname):
            signal, sr = lb.core.load(folder_train+f, sr = None)
            mask = envelope(signal, sr, 0.0005)
            wf.write(filename=clean_train+f, rate=sr, data=signal[mask])

if __name__ == "__main__":
    main()