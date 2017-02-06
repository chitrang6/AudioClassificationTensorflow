import sys
import os
import csv
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa
from matplotlib.pyplot import specgram
import tensorflow as tf
from visualize import *
from feat_ext import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.svm import SVC



#from scikits.audiolab import flacread
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve



def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def parabolic(f, x):
    # Requires real division.  Insert float() somewhere to force it?
    xv = 1/2 * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4 * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)




def freq_from_fft(sig, fs):
    # Compute Fourier transform of windowed signal
    windowed = sig * blackmanharris(len(sig))
    f = rfft(windowed)
    
    # Find the peak and interpolate to get a more accurate peak
    i = argmax(abs(f)) # Just use this for less-accurate, naive version
    true_i = parabolic(log(abs(f)), i)[0]
    
    # Convert to equivalent frequency
    return fs * true_i / len(windowed)

cwd = os.getcwd()
Train_dataset_path = ("%s/data/train")%cwd
Train_dataset, Total_Instances = load_instances(Train_dataset_path)

features, labels = np.empty((0,1)), np.empty(0)

for i in xrange (Total_Instances):
    audio_data_frame = Train_dataset[i].audio
    audio_data_rate = Train_dataset[i].rate
    gradient = np.gradient(audio_data_frame)
    funda = freq_from_fft(gradient , audio_data_rate)
    StdDev , RMSE, mfccs, chroma, mel, contrast, Abs_avg_amp, Abs_total_amp, ignore1, ignore2 = ExtractFeat(audio_data_frame, audio_data_rate)
    ext_features = np.hstack([funda])
    features = np.vstack([features,ext_features])
    print 'SpecGram Type of touch:  %s'%(Train_dataset[i].info['classlabel'])
    if Train_dataset[i].info['classlabel'] == 'pad':
        label = 0
        labels = np.append(labels , label)
    else:
        label = 1
        labels = np.append(labels , label)

print features.shape
print labels.shape
labels = np.array(labels, dtype = np.int)
#X , y = shuffle(features, labels, random_state=0)





print "One Hot label:"

#y = np.array([y, -(y-1)]).T


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=22)

clf = SVC()
SVC(C= 1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=True, random_state=12, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(X_train, y_train)
print clf.score(X_train, y_train)
y_pred = clf.predict(X_test)
print "Classification Report:"
print metrics.classification_report(y_test, y_pred)
print "Confusion Matrix:"
print metrics.confusion_matrix(y_test, y_pred)
