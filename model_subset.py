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

class Config:
    def __init__(self, mode='conv', nfilt=26, nfeat=13, nfft=512, sr=16000):
        self.mode=mode
        self.nfilt=nfilt
        self.nfeat=nfeat
        self.nfft=nfft
        self.sr=sr
        self.step=int(sr/10)
        
def build_features():
    X = []
    y = []
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p=prob_dist)
        file = np.random.choice(train_df[train_df.instrument==rand_class].index)
        audio_data, sr = lb.core.load(clean_datapath+file, sr = None)
        label = train_df.at[file, 'instrument']
        rand_idx = np.random.randint(0, audio_data.shape[0]-config.step)
        sample = audio_data[rand_idx:rand_idx+config.step]
        #X_sample = mfcc(sample, sr, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        X_sample = lb.feature.mfcc(y=sample, sr=sr, n_fft=config.nfft, n_mfcc=config.nfeat, n_mels=config.nfilt)
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    y = to_categorical(y, num_classes = len(classes))
    return X, y

def get_conv_model():
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
    model.add(Dense(units = 11, activation = tf.nn.softmax, kernel_regularizer=regularizers.l2(0.0001)))

    model.compile(optimizer=Adam(lr = 0.001), loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

clean_datapath = './clean_train/'
filenames = os.listdir(clean_datapath)

classes = list(np.unique([filename.split('_')[0] for filename in filenames]))

audio_list = []
for filename in filenames:
    instrument = filename.split('_')[0]
    value_dict = {'fname': filename, 'instrument': instrument, 'instrument_idx':classes.index(instrument)}
    audio_list.append(value_dict)

# Create a DataFrame
train_df = pd.DataFrame(audio_list)
train_df.set_index('fname', inplace=True)

for f in train_df.index:
    signal, sr = lb.core.load(clean_datapath+f)
    train_df.at[f, 'length'] = signal.shape[0]/sr
    
class_dist = train_df.groupby(['instrument'])['length'].mean()
n_samples = 2 * int(train_df['length'].sum()/0.1)
prob_dist = class_dist / class_dist.sum()

fig, ax = plt.subplots()
ax.set_title('Class distribution', y = 1.08)
ax.pie(class_dist, labels=class_dist.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode='conv')

if config.mode == 'conv':
    X_train, y_train = build_features()
    y_flat = np.argmax(y_train, axis=1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = get_conv_model()
    
class_weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
mcp = tf.keras.callbacks.ModelCheckpoint("my_model.h5", monitor="val_accuracy",
                      save_best_only=True, save_weights_only=True)

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, shuffle=True, class_weight=class_weight, verbose=1, callbacks=[mcp, es])

