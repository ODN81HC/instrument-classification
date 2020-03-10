# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 12:30:59 2020

@author: Duc An Ton
"""
import tarfile
import os
import librosa as lb
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from python_speech_features import mfcc, logfbank
from scipy.io import wavfile as wf
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, sr=16000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.sr=sr
        self.step=int(sr/10)
        
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

# Function to create DataFrame
def create_dataframe(filepath: str, classes: list):
    # Define some variables
    filenames = os.listdir(filepath)
    audio_list = []
    for filename in filenames:
        instrument = filename.split('_')[0]
        value_dict = {'fname':filename, 'instrument':instrument, 'instrument_idx':classes.index(instrument)}
        audio_list.append(value_dict)
        
    df = pd.DataFrame(audio_list)
    df.set_index('fname', inplace=True)
    # Add a column of length of the audio
    for f in tqdm(df.index):
        signal, sr = lb.core.load(filepath+f, sr = None)
        df.at[f, 'length'] = signal.shape[0]/sr
    return df

def create_dist(df):
    class_dist = df.groupby(['instrument'])['length'].mean()
    prob_dist = class_dist / class_dist.sum()
    return class_dist, prob_dist

def build_features(n_samples: int, prob_dist, df, filepath: str, config, class_dist, classes: list):
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        audio_file = np.random.choice(df[df.instrument==rand_class].index)
        audio_data, sr = lb.core.load(filepath+audio_file, sr = None)
        if audio_data.shape[0] >= config.step:
            rand_idx = np.random.randint(0, audio_data.shape[0]-config.step) # Therefore when taking the "sample", we don't run out of data
            sample = audio_data[rand_idx:rand_idx+config.step] # Take 1/10 of a second of the audio file
            #X_sample = mfcc(sample, sr, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            X_sample = lb.feature.mfcc(y=sample, sr=sr, n_fft=config.nfft, n_mfcc=config.nfeat, n_mels=config.nfilt)
            _min = min(np.amin(X_sample), _min)
            _max = max(np.amax(X_sample), _max)
            X.append(X_sample)
            y.append(classes.index(rand_class))
    X, y = np.array(X), np.array(y)
    
    # Normalize the X
    X = (X - _min) / (_max - _min)
    # Turn the X so that it has 4 dimensions
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    # One-hot encoding
    y = to_categorical(y, num_classes = len(classes))
    
    return X, y, _min, _max

def get_conv_model(input_shape, classes: list):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (3,3), strides = (1, 1),  padding="same", activation = tf.nn.relu, input_shape=input_shape))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1, 1),  padding="same", activation = tf.nn.relu))
    model.add(Conv2D(filters = 128, kernel_size = (3,3), strides = (1, 1),  padding="same", activation = tf.nn.relu))
    model.add(Conv2D(filters = 64, kernel_size = (3,3), strides = (1, 1),  padding="same", activation = tf.nn.relu))
    model.add(Conv2D(filters = 32, kernel_size = (3,3), strides = (1, 1),  padding="same", activation = tf.nn.relu))
    model.add(MaxPool2D(pool_size = (2,2), strides = None, padding = "valid"))
    model.add(Dropout(0.45))
    model.add(Flatten())
    model.add(Dense(units = 1028, activation = tf.nn.relu, kernel_regularizer=regularizers.l2(0.00025)))
    model.add(Dense(units = 514, activation = tf.nn.relu, kernel_regularizer=regularizers.l2(0.0003)))
    model.add(Dense(units = len(classes), activation = tf.nn.softmax, kernel_regularizer=regularizers.l2(0.0001)))

    model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def build_prediction(audio_dir, classes: list, config, model, _min, _max):
    y_true = []
    y_pred = []
    fn_prob = {}

    for fn in tqdm(os.listdir(audio_dir)):
        audio_data, sr = lb.core.load(os.path.join(audio_dir, fn), sr = None)
        mask = envelope(audio_data, sr, 0.0005)
        audio_data = audio_data[mask]
        
        label = fn.split('_')[0]
        c = classes.index(label)
        y_prob = []
        
        for i in range(0, audio_data.shape[0]-config.step, config.step):
            sample = audio_data[i:i+config.step]
            #x = mfcc(sample, sr, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
            x = lb.feature.mfcc(y=sample, sr=sr, n_fft=config.nfft, n_mfcc=config.nfeat, n_mels=config.nfilt)
            x = (x - _min)/ (_max - _min)
            
            if config.mode == 'conv':
                x = x.reshape(1, x.shape[0], x.shape[1], 1)
            
            y_hat = model.predict(x)
            y_prob.append(y_hat)
            y_pred.append(np.argmax(y_hat))
            y_true.append(c)
        
        fn_prob[fn] = np.mean(y_prob, axis = 0).flatten()
        
    return y_true, y_pred, fn_prob
            
            
# Define the training data path
train_datapath = './clean_train/'
valid_datapath = './clean_valid/'
test_datapath = './wavfile_test/'

train_filenames = os.listdir(train_datapath)
classes = list(np.unique([filename.split('_')[0] for filename in train_filenames]))

# Create a training dataFrame
#train_df = create_dataframe(filepath = train_datapath, classes = classes)
#valid_df = create_dataframe(filepath = valid_datapath, classes = classes)
#
## The dataframe is saved into csv file
#train_df.to_csv('train.csv', index=False)
#valid_df.to_csv('valid.csv', index=False)

# Load csv file
train_df = pd.read_csv('train.csv')
train_df.set_index('fname', inplace=True)
valid_df = pd.read_csv('valid.csv')
valid_df.set_index('fname', inplace=True)

# Some param for train and valid respectively
train_class_dist, train_prob_dist = create_dist(train_df)
train_n_samples = int(train_df['length'].sum()/3)

valid_class_dist, valid_prob_dist = create_dist(valid_df)
valid_n_samples = int(valid_df['length'].sum()/0.1)

config = Config(mode='conv')

if config.mode == 'conv':
    X_train, y_train, train_min, train_max = build_features(train_n_samples, train_prob_dist, train_df, train_datapath, config, train_class_dist, classes)
    # Save the X_train and y_train because it takes a lot of times to run this again
    np.save('X_train.npy', X_train)
    np.save('y_train.npy', y_train)
    np.save('train_min.npy', train_min)
    np.save('train_max.npy', train_max)
    # Load them again to run the model
#    X_train = np.load('X_train.npy')
#    y_train = np.load('y_train.npy')
#    train_max = np.load('train_max.npy')
#    train_min = np.load('train_min.npy')
    
    X_valid, y_valid, _, _ = build_features(valid_n_samples, valid_prob_dist, valid_df, valid_datapath, config, valid_class_dist, classes)
    y_flat = np.argmax(y_train, axis=1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = get_conv_model(input_shape, classes)
    
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)

mcp = tf.keras.callbacks.ModelCheckpoint("my_model.h5", monitor="val_accuracy",
                      save_best_only=True, save_weights_only=True)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 8)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.45,
                              patience=2, min_lr=0.000001, verbose=1)

#model.fit(X_train, y_train, epochs=32, batch_size=32, shuffle=True, validation_data=(X_valid, y_valid), class_weight=class_weight, verbose=1, callbacks=[mcp, es, reduce_lr])
model.fit(X_train, y_train, epochs=32, batch_size=32, shuffle=True, validation_split=0.2, class_weight=class_weight, verbose=1, callbacks=[mcp, es, reduce_lr])

# Make predictions
y_true, y_pred, fn_prob = build_prediction(audio_dir=test_datapath, classes = classes, config=config, model = model, _min = train_min, _max = train_max)
acc_score = accuracy_score(y_true = y_true, y_pred=y_pred)
print("The accuracy score for the task is: {}".format(acc_score))

# Create a data frame for the test values
dict_list = []
for f, probs in fn_prob.items():
    value_dict = {'fname':f, 'true_class':f.split('_')[0], 'pred_class':classes[np.argmax(probs)]}
    dict_list.append(value_dict)
test_df = pd.DataFrame(dict_list)
test_df.set_index('fname', inplace=True)
# Extract to csv file
test_df.to_csv('test_df.csv', index=False)
