import sys
import os
import csv
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa
from matplotlib.pyplot import specgram
import tensorflow as tf
from utility import *
import math



class FeatExtraction(object):
    """This class contains the necessory methods related to the Feature Extraction."""
    def __init__(self, NMFCC , NMEL ,ROLL_percent, Audio_data, Audio_rate):
        self.nmfcc = NMFCC
        self.ROLL_percent = ROLL_percent
        self.data = Audio_data
        self.rate = Audio_rate
        self.nmale = NMEL


    def ExtractFeat(self):
        Stft = librosa.stft(self.data)
        Stft = np.abs(Stft)
        Spectogram, phase = librosa.magphase(Stft)
        StdDev = np.std(np.abs(self.data))
        #print "StdDev: "
        #print StdDev
        RMSE = np.mean(librosa.feature.rmse(y=self.data, S=Spectogram, n_fft=2048).T,axis = 0)
        #print "RMSE: "
        #print RMSE
        mfccs = np.mean(librosa.feature.mfcc(y=self.data, sr=self.rate, n_mfcc=self.nmfcc).T,axis=0)
        #print "MFCCS: "
        #print mfccs.shape
        chroma = np.mean(librosa.feature.chroma_stft(S=Stft, sr=self.rate).T, axis = 0)
        #print "Chroma"
        #print chroma.shape
        S = librosa.feature.melspectrogram(self.data, sr=self.rate, n_mels=self.nmale)
        log_mel = np.mean(librosa.logamplitude(S, ref_power=np.max).T, axis = 0)
        mel = np.mean(librosa.feature.melspectrogram(self.data, sr=self.rate).T , axis = 0)
        #print "Mel: "
        #print mel.shape
        contrast = np.mean(librosa.feature.spectral_contrast(S=Stft, sr=self.rate).T, axis = 0)
        #tonnetz = librosa.feature.tonnetz(y=self.data, sr=self.rate)
        #print "contrast: "
        #print contrast.shape
        Abs_avg_amp = np.average(np.abs(self.data))
        #print "Abs Avf ampli: "
        #print Abs_avg_amp
        Abs_total_amp = np.sum(np.abs(self.data))
        if_gram, D = librosa.ifgram(self.data)
        centroid = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
        roll_off = librosa.feature.spectral_rolloff(y=self.data, sr=self.rate, roll_percent= self.ROLL_percent)
        #print "Abs Total ampli: "
        #print Abs_total_amp
        #print roll_off
        #print centroid
        S = librosa.feature.melspectrogram(self.data, sr=self.rate, n_mels=self.nmfcc)
        log_S = np.mean(librosa.logamplitude(S, ref_power=np.max).T , axis=0)
        mfcc_for_delta = librosa.feature.mfcc(S=log_S, n_mfcc=self.nmfcc)

        delta_mfcc  = np.mean(librosa.feature.delta(mfcc_for_delta).T , axis = 0)
        delta2_mfcc = np.mean(librosa.feature.delta(mfcc_for_delta, order=2).T , axis = 0)

        #print delta_mfcc.shape
        #print delta2_mfcc.shape
        return StdDev, RMSE, mfccs, chroma, log_mel, contrast, Abs_avg_amp, Abs_total_amp, centroid[0][0] , roll_off[0][0], delta_mfcc, delta2_mfcc


