from scipy.io import wavfile


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    assert len(signal.shape) == 1, "Only Support Mono Wav File!"
    return fs, signal


def write_wav(fname, fs, signal):
    wavfile.write(fname, fs, signal)


def monophonic(signal):
    if signal.ndim > 1:
        signal = signal[:, 0]
    return signal

