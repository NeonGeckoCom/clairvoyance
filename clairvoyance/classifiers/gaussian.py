import pickle as pickle
import traceback as tb
from collections import defaultdict
from clairvoyance.filters.VAD import VAD
from clairvoyance.features import extract_features
import operator
import numpy as np
from sklearn.mixture import GaussianMixture


class GMMSet:
    def __init__(self, gmm_order=32):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(x)
        self.gmms.append(gmm)
        return len(self.y)

    def refit(self, x, idx):
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(x)
        self.gmms[idx] = gmm

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    def predict_label(self, x):
        scores = self.predict_scores(x)
        result = [(self.y[index], value) for (index, value) in
                  enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        return p[0]

    def predict_all(self, x):
        scores = [self.gmm_score(gmm, x) // len(x) for gmm in self.gmms]
        result = [(self.y[index], value) for (index, value) in
                  enumerate(scores)]
        results = {}
        for label, score in result:
            results[label] = score
        return results

    def predict(self, x):
        scores = [self.gmm_score(gmm, x) // len(x) for gmm in self.gmms]
        result = [(self.y[index], value) for (index, value) in
                  enumerate(scores)]
        p = max(result, key=operator.itemgetter(1))
        return p

    def predict_scores(self, x):
        scores = [self.gmm_score(gmm, x) // len(x) for gmm in self.gmms]
        return scores


class ModelInterface:
    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()
        self.gmm_map = {}
        self.vad = VAD()

    def init_noise(self, fs, signal):
        """
        init vad from environment noise
        """
        self.vad.init_noise(fs, signal)

    def filter(self, fs, signal):
        """
        use VAD (voice activity detection) to filter out silence part of a signal
        """
        ret, intervals = self.vad.filter(fs, signal)
        orig_len = len(signal)

        if len(ret) > orig_len // 3:
            # signal is filtered by VAD
            return ret
        return np.array([])

    def enroll(self, name, fs, signal):
        """
        add the signal to this person's training dataset
        name: person's name
        """
        feat = extract_features((fs, signal))
        self.features[name].extend(feat)

    def train(self):
        self.gmmset = GMMSet()
        for name, feats in self.features.items():
            idx = self.gmmset.fit_new(feats, name)
            self.gmm_map[idx] = name

    def retrain_user(self, name):
        for idx in self.gmm_map:
            if self.gmm_map[idx] == name:
                x = self.features[name]
                self.gmmset.refit(x, idx)

    def predict(self, fs, signal):
        """
        return a label (name)
        """
        try:
            feat = extract_features((fs, signal))
        except Exception as e:
            print(tb.format_exc())
            return None
        return self.gmmset.predict(feat)

    def predict_scores(self, fs, signal):
        """
        return scores
        """
        try:
            feat = extract_features((fs, signal))
        except Exception as e:
            print(tb.format_exc())
            return None
        return self.gmmset.predict_scores(feat)

    def dump(self, fname):
        """ dump all models to file"""
        with open(fname, 'w') as f:
            pickle.dump(self, f, -1)

    @staticmethod
    def load(fname):
        """ load from a dumped model file"""
        with open(fname, 'r') as f:
            return pickle.load(f)
