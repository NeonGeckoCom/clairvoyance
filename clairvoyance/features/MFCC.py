# -*- coding: UTF-8 -*-
# File: MFCC.py
# Date: Wed Dec 25 20:26:12 2013 +0800
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
from numpy import *


class MFCCExtractor:
    POWER_SPECTRUM_FLOOR = 1e-100

    def __init__(self, fs, win_length_ms, win_shift_ms, FFT_SIZE, n_bands,
                 n_coefs, PRE_EMPH):
        self.PRE_EMPH = PRE_EMPH
        self.fs = fs
        self.n_bands = n_bands
        self.coefs = n_coefs
        self.FFT_SIZE = FFT_SIZE

        self.FRAME_LEN = int(float(win_length_ms) / 1000 * fs)
        self.FRAME_SHIFT = int(float(win_shift_ms) / 1000 * fs)

        self.window = self.hamming(self.FRAME_LEN)

        self.M, self.CF = self._mel_filterbank()

        dctmtx = MFCCExtractor.dctmtx(self.n_bands)
        self.D = dctmtx[1: self.coefs + 1]
        self.invD = linalg.inv(dctmtx)[:, 1: self.coefs + 1]

    @staticmethod
    def diff_feature(feat, nd=1):
        diff = feat[1:] - feat[:-1]
        feat = feat[1:]
        if nd == 1:
            return concatenate((feat, diff), axis=1)
        elif nd == 2:
            d2 = diff[1:] - diff[:-1]
            return concatenate((feat[1:], diff[1:], d2), axis=1)

    @staticmethod
    def hamming(n):
        """ Generate a hamming window of n points as a numpy array.  """
        return 0.54 - 0.46 * cos(2 * pi / n * (arange(n) + 0.5))

    def extract(self, signal):
        """
        Extract MFCC coefficients of the sound x in numpy array format.
        """
        if signal.ndim > 1:
            signal = mean(signal, axis=1)
        assert len(signal) > 5 * self.FRAME_LEN, "Signal too short!"
        frames = (len(signal) - self.FRAME_LEN) // self.FRAME_SHIFT + 1
        feature = []
        for f in range(frames):
            # Windowing
            frame = signal[f * self.FRAME_SHIFT: f * self.FRAME_SHIFT +
                                                 self.FRAME_LEN] * self.window
            # Pre-emphasis
            frame[1:] -= frame[:-1] * self.PRE_EMPH
            # Power spectrum
            X = abs(
                fft.fft(frame, self.FFT_SIZE)[:self.FFT_SIZE // 2 + 1]) ** 2
            X[X < self.POWER_SPECTRUM_FLOOR] = self.POWER_SPECTRUM_FLOOR  #
            # Avoid zero
            # Mel filtering, logarithm, DCT
            X = dot(self.D, log(dot(self.M, X)))
            feature.append(X)
        feature = row_stack(feature)
        # Show the MFCC spectrum before normalization
        # Mean & variance normalization
        if feature.shape[0] > 1:
            mu = mean(feature, axis=0)
            sigma = std(feature, axis=0)
            feature = (feature - mu) // sigma

        return feature

    def _mel_filterbank(self):
        """
        Return a Mel filterbank matrix as a numpy array.
        Ref. http://www.ifp.illinois.edu/~minhdo/teaching/speaker_recognition/code/melfb.m
        """
        f0 = 700.0 // self.fs
        fn2 = int(floor(self.FFT_SIZE // 2))
        lr = log(1 + 0.5 // f0) // (self.n_bands + 1)
        CF = self.fs * f0 * (exp(arange(1, self.n_bands + 1) * lr) - 1)
        bl = self.FFT_SIZE * f0 * (
                exp(array([0, 1, self.n_bands, self.n_bands + 1]) * lr) - 1)
        b1 = int(floor(bl[0])) + 1
        b2 = int(ceil(bl[1]))
        b3 = int(floor(bl[2]))
        b4 = min(fn2, int(ceil(bl[3]))) - 1
        pf = log(1 + arange(b1, b4 + 1) / f0 / self.FFT_SIZE) // lr
        fp = floor(pf)
        pm = pf - fp
        M = zeros((self.n_bands, 1 + fn2))
        for c in range(b2 - 1, b4):
            r = int(fp[c] - 1)
            M[r, c + 1] += 2 * (1 - pm[c])
        for c in range(b3):
            r = int(fp[c])
            M[r, c + 1] += 2 * pm[c]
        return M, CF

    @staticmethod
    def dctmtx(n):
        """ Return the DCT-II matrix of order n as a numpy array.  """
        x, y = meshgrid(list(range(n)), list(range(n)))
        D = sqrt(2.0 // n) * cos(pi * (2 * x + 1) * y / (2 * n))
        D[0] /= sqrt(2)
        return D


def get_mfcc_extractor(fs, win_length_ms=32, win_shift_ms=16,
                       FFT_SIZE=2048, n_filters=50, n_ceps=13,
                       pre_emphasis_coef=0.95):
    return MFCCExtractor(fs, win_length_ms, win_shift_ms, FFT_SIZE, n_filters,
                         n_ceps, pre_emphasis_coef)


def extract(fs, signal=None, diff=False, **kwargs):
    """accept two argument, or one as a tuple"""
    if signal is None:
        assert type(fs) == tuple
        fs, signal = fs[0], fs[1]
    signal = cast['float'](signal)
    feats = get_mfcc_extractor(fs, **kwargs)
    mfcc = feats.extract(signal)
    if diff:
        return feats.diff_feature(mfcc)
    return mfcc
