# Clairvoyance

speaker recognition made easy


## Usage

```python
from clairvoyance import Clairvoyance

train, test = get_data()
# {"speaker_name": ["path_to.wav"]

recognition = Clairvoyance()

speakers = {}
for speaker in train:
    wav_files = train[speaker]
    speakers[speaker] = recognition.get_speaker_encoding(wav_files)

for speaker in test:
    for utterance in test[speaker]:
        test_embed = recognition.get_speaker_encoding(utterance)
        for speaker in speakers:
            speaker_embed = speakers[speaker]
            score = recognition.speaker_similarity(test_embed, speaker_embed)
            
            if score >= 0.75:
                print("speaker is", speaker)
            else:
                print("unknown speaker")

```
