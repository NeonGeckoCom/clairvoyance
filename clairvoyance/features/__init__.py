from clairvoyance.features import MFCC


def extract_features(tup):
    mfcc = MFCC.extract(tup)
    # concat more features here
    return mfcc
