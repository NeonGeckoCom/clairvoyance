from resemblyzer import VoiceEncoder, preprocess_wav
from clairvoyance.exceptions import InvalidFileFormat
from os.path import isfile
import numpy as np


class Clairvoyance:
    def __init__(self):
        self.encoder = VoiceEncoder("cpu")

    def get_speaker_encoding(self, wav_files):
        if not isinstance(wav_files, list):
            wav_files = [wav_files]
        encoding = []
        for wav_file in wav_files:
            if not isfile(wav_file):
                raise FileNotFoundError
            if not wav_file.endswith(".wav") \
                    and not wav_file.endswith(".flac") \
                    and not wav_file.endswith(".mp3"):
                raise InvalidFileFormat
            wav = preprocess_wav(wav_file)
            encoding += [self.encoder.embed_utterance(wav)]
        return encoding

    def speaker_similarity(self, embed1, embed2):
        # Compute the similarity matrix. The similarity of two embeddings is simply their dot
        # product, because the similarity metric is the cosine similarity and the embeddings are
        # already L2-normed.
        score = np.inner(embed1, embed2)
        return np.average(score)
