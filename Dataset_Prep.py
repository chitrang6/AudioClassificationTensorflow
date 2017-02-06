import sys
import os
import csv
import numpy as np
from utility import *
from Feat_Extraction import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math




#from scikits.audiolab import flacread
from numpy.fft import rfft, irfft
from numpy import argmax, sqrt, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve



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
    return fs * true_i / len(windowed)

def running_mean(x, windowSize):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize


__NUM_OF_MFCCSPOINTS = 32
__NUM_OF_MELPOINTS = 32

__ROLL_OF_PERCENT = 0.50



cutOffFrequency = 500.0



def GetDatasetSplit():
	cwd = os.getcwd()
	Train_dataset_path = ("%s/data/train")%cwd
	Train_dataset, Total_Instances = load_instances(Train_dataset_path)
	features, labels = np.empty((0,65)), np.empty(0)

	for i in xrange (Total_Instances):
		audio_data_frame = Train_dataset[i].audio
		audio_data_rate = Train_dataset[i].rate
		freqRatio = (cutOffFrequency/audio_data_rate)
		windowSize = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)
		#audio_data_frame = running_mean(audio_data_frame, windowSize)
		gradient = np.gradient(audio_data_frame)
		fundamenta_freq = freq_from_fft(gradient , audio_data_rate)
		major_axis = Train_dataset[i].touch['major'] 
		minor_axis = Train_dataset[i].touch['minor'] 
		area = math.pi * major_axis * minor_axis
		Feat_instance = FeatExtraction(__NUM_OF_MFCCSPOINTS , __NUM_OF_MELPOINTS,  __ROLL_OF_PERCENT , audio_data_frame , audio_data_rate)
		StdDev, RMSE, mfccs, chroma, mel, contrast, Abs_avg_amp, Abs_total_amp, centroid , roll_off, delta_mfcc, delta2_mfcc = Feat_instance.ExtractFeat()
		#print mfccs.shape 
		#print chroma.shape 
		#print mel.shape 
		#print contrast.shape
		#print delta_mfcc.shape 
		#print delta2_mfcc.shape 
		ext_features = np.hstack([mel, mfccs , fundamenta_freq])
		features = np.vstack([features,ext_features])
		print 'Type of touch:  %s'%(Train_dataset[i].info['classlabel'])
		if Train_dataset[i].info['classlabel'] == 'pad':
			label = 0
			labels = np.append(labels , label)
		else:
			label = 1
			labels = np.append(labels , label)


	labels = np.array(labels, dtype = np.int)
	X , y = shuffle(features, labels, random_state=0)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
	y_train = np.array([y_train, -(y_train-1)]).T
	y_test = np.array([y_test, -(y_test-1)]).T

	return X_train, X_test, y_train, y_test


def GetFinalTestData():
	cwd = os.getcwd()
	Test_dataset_path = ("%s/data/test")%cwd
	Test_dataset, Total_Instances = load_instances(Test_dataset_path)
	features, labels = np.empty((0,65)), np.empty(0)

	for i in xrange (Total_Instances):
		audio_data_frame = Test_dataset[i].audio
		audio_data_rate = Test_dataset[i].rate
		freqRatio = (cutOffFrequency/audio_data_rate)
		windowSize = int(math.sqrt(0.196196 + freqRatio**2)/freqRatio)
		#audio_data_frame = running_mean(audio_data_frame, windowSize)
		gradient = np.gradient(audio_data_frame)
		fundamenta_freq = freq_from_fft(gradient , audio_data_rate)
		major_axis = Test_dataset[i].touch['major'] 
		minor_axis = Test_dataset[i].touch['minor'] 
		area = math.pi * major_axis * minor_axis
		Feat_instance = FeatExtraction(__NUM_OF_MFCCSPOINTS , __NUM_OF_MELPOINTS,  __ROLL_OF_PERCENT , audio_data_frame , audio_data_rate)
		StdDev, RMSE, mfccs, chroma, mel, contrast, Abs_avg_amp, Abs_total_amp, centroid , roll_off, delta_mfcc, delta2_mfcc = Feat_instance.ExtractFeat()
		#print mfccs.shape 
		#print chroma.shape 
		#print mel.shape 
		#print contrast.shape
		#print delta_mfcc.shape 
		#print delta2_mfcc.shape 
		ext_features = np.hstack([mel, mfccs , fundamenta_freq])
		features = np.vstack([features,ext_features])
		
	X = shuffle(features, random_state=0)
	return X












