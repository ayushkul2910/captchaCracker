#!/usr/bin/env python3

import librosa
import numpy as np
import operator
from pydub import AudioSegment
from pydub.silence import split_on_silence
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

class CaptchaCracker:

    letterList = []
    RESOURCE_DIR = "./resources"
    DATA_DIR = "/data/"

    def __init__(self):
        self.letterList = ["2.npy", "5.npy", "8.npy", "b.npy", "e.npy", "h.npy", "m.npy", "r.npy", "u.npy", "y.npy", "3.npy", "6.npy", "9.npy", "c.npy", "f.npy", "k.npy", "n.npy", "s.npy", "v.npy", "z.npy", "4.npy", "7.npy", "a.npy", "d.npy", "g.npy", "l.npy", "p.npy", "t.npy", "w.npy"]


    def audioSegmentToArray(self, audiosegment):
        samples = audiosegment.get_array_of_samples()
        samples_float = librosa.util.buf_to_float(samples,n_bytes=2, dtype=np.float32)
        if audiosegment.channels==2:
            sample_left= np.copy(samples_float[::2])
            sample_right= np.copy(samples_float[1::2])
            sample_all = np.array([sample_left,sample_right])
        else:
            sample_all = samples_float
        return [sample_all,audiosegment.frame_rate]


    def getArrayDistance(self, sound_array, numpy_array):
        m1 = librosa.feature.mfcc(sound_array[0],sound_array[1])
        a,_ = fastdtw(m1.T, numpy_array, dist = euclidean)
        return a


    def audioToText(self, audioFilePath):
        text = ''
        letterDist = {}
        sound = AudioSegment.from_wav(audioFilePath)
        chunks = split_on_silence(sound, min_silence_len = 50, silence_thresh = -20, keep_silence = 100)
        count=0
        for chunk in chunks:
            for i in self.letterList:
                sound_array = np.load(self.RESOURCE_DIR + self.DATA_DIR + i)
                letterDist[i] = self.getArrayDistance(self.audioSegmentToArray(chunk), sound_array)
            count+=1
            letter = min(letterDist.items(), key = operator.itemgetter(1))[0]
            letter = letter[0:1]
            text+=letter
            letterDist = {}
        if len(text) == 7 and 'h' in text:
            textList = list(text)
            textList.remove('h')
            return ''.join(textList)
        return text