#!/usr/bin/env python
""" Functions for the preprocessing of the audio wave files. """

import sys
import os
import csv
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa
from matplotlib.pyplot import specgram


def SpecGramPlot(Sound_frame):
    specgram(np.array(Sound_frame), Fs = 96000)
    plt.show()

def Log_PowerSpecGramPlot(Sound_frame):
     Log_data = librosa.logamplitude(np.abs(librosa.stft(Sound_frame))**2, ref_power=np.max)
     librosa.display.specshow(Log_data,x_axis='time' ,y_axis='log')
     plt.show()



class Instance(object):
    """
        Instance class represents set of raw data collected per data instance
    """
    def __init__(self, dir):
        self.audio, self.rate = self._load_audio(dir)
        self.touch = self._load_touch_data(dir)
        self.info = self._load_instance_info(dir)

    def _load_audio(self, dir):
        """ load audio data
            param dir: a path to an instance directory
            return: audio data
        """
        rate, wav = scipy.io.wavfile.read(os.path.join(dir, "audio.wav"))
        #y, sr = librosa.load(os.path.join(dir, "audio.wav"))
        return wav , rate

    def _load_touch_data(self, dir):
        """ load touch data
            param dir: a path to an instance directory
            return : a dictionary contains touch data
        """
        with open(os.path.join(dir, "touch.csv"), "rU") as f:
            reader = csv.DictReader(f)
            for touch in reader:
                for key in  touch.keys():
                    touch[key] = float(touch[key])
                break
        return touch

    def _load_instance_info(self, dir):
        """ load instance info from a directory path
            param dir: a path to an instance directory
            return: a dictionary contains basic instance information
        """
        info = {}
        user_dirnames = os.path.basename(os.path.dirname(dir)).split("-")
        info["surface"] = user_dirnames[0]
        info["user"] = user_dirnames[1]
        instance_dirnames = os.path.basename(dir).split("-")
        info["timestamp"] = instance_dirnames[0]
        # set None to classlabel if it's test data
        info["classlabel"] = instance_dirnames[1] if len(instance_dirnames) == 2 else None
        return info


def load_instances(dir):
    """ function for loading raw data instances
        param dir: a path to a data directory (i.e. task_data/train or task_data/test)
        return: a list of data instance objects
    """
    print "Preparing the Trainig dataset..."
    Total_Examples = 0
    instances = []
    for root, dirs, files in os.walk(os.path.join(dir)):
        for filename in files:
            if filename == "audio.wav":
                Total_Examples = Total_Examples +1
                instances.append(Instance(root))
    return instances, Total_Examples






#cwd = os.getcwd()
#Train_dataset_path = ("%s/data/train")%cwd

#Train_dataset, Total_Examples = load_instances(Train_dataset_path)

#print Train_dataset[0].info
#Num = 19000

#_SAM_FREQ = 96000


#for i in range(5):
#    shape = Train_dataset[Num].audio.shape
    #print shape
    #print Train_dataset[Num].rate
#    audio_data_frame = Train_dataset[Num].audio
#    frate = Train_dataset[Num].rate
#    length = len(audio_data_frame)
#    plt.plot(audio_data_frame,'r')
#    plt.title('Type of touch:  %s'%(Train_dataset[Num].info['classlabel']))
#    plt.show()
#    w = np.fft.fft(audio_data_frame , n = 4096)
    #print np.abs(w)
#    freqs = np.fft.fftfreq(len(w))
#    print(freqs.min(), freqs.max())
#    idx = np.argmax(np.abs(w))
#    freq = freqs[idx]
#    freq_in_hertz = abs(freq * frate)
#    print(freq_in_hertz)
#    hello = dspUtil.calculateFFT(audio_data_frame, frate, 16)
#    print hello
#    plt.title('SpecGram Type of touch:  %s'%(Train_dataset[Num].info['classlabel']))
#    SpecGramPlot(audio_data_frame)
#    plt.title('SpecGram Log Type of touch:  %s'%(Train_dataset[Num].info['classlabel']))
#    Log_PowerSpecGramPlot(audio_data_frame)
#    Num = Num + 1
