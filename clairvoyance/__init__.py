from resemblyzer import VoiceEncoder, preprocess_wav
from clairvoyance.exceptions import InvalidFileFormat
from os.path import isfile, join
from os import listdir
import numpy as np


class Clairvoyance:
    def __init__(self, gpu=False):
        if gpu:
            self.encoder = VoiceEncoder()
        else:
            self.encoder = VoiceEncoder("cpu")
        self.speakers = {}

    @property
    def speaker_ids(self):
        return list(self.speakers.keys())

    def forget_speakers(self):
        """ clear list of enrolled speakers """
        self.speakers = {}

    def forget_speaker(self, speaker_id):
        """ forget speaker with speaker_id """
        if speaker_id in self.speaker_ids:
            self.speakers.pop(speaker_id)

    def enroll_speaker(self, speaker_id, sound_path):
        """ add sound file encoding to known speakers with speaker_id """
        embed = self.get_speaker_encoding(sound_path)
        self.speakers[speaker_id] += [embed]

    def enroll_folder(self, folder_path):
        """
        given a folder structure like

           folder
           ├── speaker_1
           │   └── 1.wav
           └── speaker 2
               └── 1.wav

        load encodings for each speaker
        """
        for speaker in listdir(folder_path):
            if speaker not in self.speakers:
                self.speakers[speaker] = []
            for utterance in listdir(join(folder_path, speaker)):
                if utterance.endswith(".wav") \
                        or utterance.endswith(".flac") \
                        or utterance.endswith(".mp3"):
                    self.enroll_speaker(speaker, join(folder_path, speaker, utterance))

    def closest_speaker(self, sound_path):
        """
        calculate speaker similarity to enrolled speakers
        return best speaker, score  (tuple)
        """
        test_embed = self.get_speaker_encoding(sound_path)
        best_score = 0
        best_speaker = "unknown"
        for speaker in self.speakers:
            speaker_embed = self.speakers[speaker]
            score = self.speaker_similarity(test_embed, speaker_embed)
            if score > best_score:
                best_score = score
                best_speaker = speaker
        return best_speaker, best_score

    def predict_speaker(self, sound_path, model):
        # TODO predict speaker using provided svm model
        raise NotImplementedError

    def train_folder(self, folder_path):
        # TODO  given a folder like
        #     # folder
        #     # ├── speaker_1
        #     # │   └── 1.wav
        #     # └── speaker 2
        #     #     └── 1.wav
        # get encodings for each .wa
        # train a svm classifier
        raise NotImplementedError

    def get_speaker_encoding(self, wav_files):
        """
        get voice encoding for a sound file or list of files
        """
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

    @staticmethod
    def speaker_similarity(embed1, embed2):
        """
        Compute the similarity matrix. The similarity of two embeddings is simply their dot
        product, because the similarity metric is the cosine similarity and the embeddings are
        already L2-normed.
        """
        score = np.inner(embed1, embed2)
        return np.average(score)
