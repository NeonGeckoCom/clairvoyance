# -*- coding: UTF-8 -*-
# File: noisered.py
# Date: Fri Dec 27 04:23:28 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
from scipy.io import wavfile
import os
from tempfile import gettempdir
from os.path import join
from random import random
from clairvoyance.utils import monophonic


class NoiseReduction:
    NOISE_WAV = join(gettempdir(), "noise.wav")
    NOISE_MODEL = join(gettempdir(), "noise.prof")
    THRESH = 0.21

    def init_noise(self, fs, signal):
        wavfile.write(self.NOISE_WAV, fs, signal)
        os.system("sox {0} -n noiseprof {1}".format(self.NOISE_WAV,
                                                    self.NOISE_MODEL))

    def filter(self, fs, signal):
        rand = random.randint(1, 100000)
        fname = join(gettempdir(), "tmp{}.wav".format(rand))
        signal = monophonic(signal)
        wavfile.write(fname, fs, signal)
        fname_clean = join(gettempdir(), "tmp{}-clean.wav".format(rand))
        os.system("sox {0} {1} noisered {2} {3}".format(fname, fname_clean,
                                                        self.NOISE_MODEL,
                                                        self.THRESH))
        fs, signal = wavfile.read(fname_clean)
        signal = monophonic(signal)

        os.remove(fname)
        os.remove(fname_clean)
        return signal
