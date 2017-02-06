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
import math





#cwd = os.getcwd()
#Train_dataset_path = ("%s/data/train")%cwd
#Train_dataset, Total_Instances = load_instances(Train_dataset_path)




# Features we want to extract.
# 1. Standard Deviation
# 2. Absolute average amplitude
# 3. Absolute total ampliture
# 4. RMSE
# 5. Roll-off frequency
# 6. Zero- Crossing rate
# 7.

def ExtractFeat(Audio_data_frame, Audio_frame_rate):

    Stft = librosa.stft(Audio_data_frame)
    Stft = np.abs(Stft)
    Spectogram, phase = librosa.magphase(Stft)
    StdDev = np.std(np.abs(Audio_data_frame))
    #print "StdDev: "
    #print StdDev
    RMSE = np.mean(librosa.feature.rmse(y=Audio_data_frame, S=Spectogram, n_fft=2048).T,axis = 0)
    #print "RMSE: "
    #print RMSE
    mfccs = np.mean(librosa.feature.mfcc(y=Audio_data_frame, sr=Audio_frame_rate, n_mfcc=128).T,axis=0)
    #print "MFCCS: "
    #print mfccs.shape
    chroma = np.mean(librosa.feature.chroma_stft(S=Stft, sr=Audio_frame_rate).T, axis = 0)
    #print "Chroma"
    #print chroma.shape
    mel = np.mean(librosa.feature.melspectrogram(Audio_data_frame, sr=Audio_frame_rate).T , axis = 0)
    #print "Mel: "
    #print mel.shape
    contrast = np.mean(librosa.feature.spectral_contrast(S=Stft, sr=Audio_frame_rate).T, axis = 0)
    #tonnetz = librosa.feature.tonnetz(y=Audio_data_frame, sr=Audio_frame_rate)
    #print "contrast: "
    #print contrast.shape
    Abs_avg_amp = np.average(np.abs(Audio_data_frame))
    #print "Abs Avf ampli: "
    #print Abs_avg_amp
    Abs_total_amp = np.sum(np.abs(Audio_data_frame))
    if_gram, D = librosa.ifgram(Audio_data_frame)
    centroid = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
    roll_off = librosa.feature.spectral_rolloff(y=Audio_data_frame, sr=Audio_frame_rate, roll_percent=0.30)
    #print "Abs Total ampli: "
    #print Abs_total_amp
    #print roll_off
    #print centroid
    #S = librosa.feature.melspectrogram(Audio_data_frame, sr=Audio_frame_rate, n_mels=128)
    #log_S = librosa.logamplitude(S, ref_power=np.max)
    #mfcc_for_delta = librosa.feature.mfcc(S=log_S, n_mfcc=128)

    #delta_mfcc  = np.mean(librosa.feature.delta(mfcc_for_delta).T , axis = 0)
    #delta2_mfcc = np.mean(librosa.feature.delta(mfcc_for_delta, order=2).T , axis = 0)

    #print delta_mfcc.shape
    #print delta2_mfcc.shape
    return StdDev, RMSE, mfccs, chroma, mel, contrast, Abs_avg_amp, Abs_total_amp, centroid[0][0] , roll_off[0][0]


#Num = 1
#for i in range(5):
#    shape = Train_dataset[Num].audio.dtype
#    print shape
#    print Train_dataset[Num].rate
#    audio_data_frame = Train_dataset[Num].audio
#    audio_data_rate = Train_dataset[Num].rate
#    length = len(audio_data_frame)
#    print Train_dataset[Num].touch
#    area_eclipse = Train_dataset[Num].touch['major'] * Train_dataset[Num].touch['minor']*math.pi
#    print area_eclipse
#    plt.plot(audio_data_frame,'r')
#    plt.title('Type of touch:  %s'%(Train_dataset[Num].info['classlabel']))
#    plt.show()
#    plt.title('SpecGram Type of touch:  %s'%(Train_dataset[Num].info['classlabel']))
#    SpecGramPlot(audio_data_frame)
#    plt.title('SpecGram Log Type of touch:  %s'%(Train_dataset[Num].info['classlabel']))
#    Log_PowerSpecGramPlot(audio_data_frame)
#    ExtractFeat(audio_data_frame, audio_data_rate)
#    Num = Num + 1
