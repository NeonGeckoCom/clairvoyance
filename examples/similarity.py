from clairvoyance import Clairvoyance
from os.path import join, dirname

recognition = Clairvoyance()

train_folder = join(dirname(__file__), "samples/train")

# load a folder of know speakers
recognition.enroll_folder(train_folder)

unk_utterance = join(dirname(__file__), "samples", "test", "user", "me1.wav")
known_utterance = join(dirname(__file__), "samples", "test", "user2", "beirao1.wav")

speaker, score = recognition.closest_speaker(known_utterance)
assert score > 0.75
assert speaker == "user2"

speaker, score = recognition.closest_speaker(unk_utterance)
assert score < 0.75
