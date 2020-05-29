import requests
from requests.exceptions import ConnectionError


class XerlokVoice:
    SPEAKER_URL = "http://0.0.0.0:5678/speaker_recognition/"

    def __init__(self, api):
        self.api = api
        self.headers = {"Authorization": str(self.api)}

    def recognize_speaker(self, wav_file):
        filepath = wav_file
        with open(filepath) as fh:
            mydata = fh.read()
        try:
            response = requests.put(
                XerlokVoice.SPEAKER_URL + "recognize",
                data=mydata,
                headers=self.headers
            )
        except ConnectionError:
            raise ConnectionError("The Speaker Recognition service is "
                                  "unavailable.")

        try:
            return response.json()
        except:
            return response.text

    def train_speaker(self, user, wav_file):
        filepath = wav_file
        with open(filepath) as fh:
            mydata = fh.read()
        try:
            response = requests.put(
                XerlokVoice.SPEAKER_URL + "train/" + user,
                data=mydata,
                headers=self.headers
            )
        except ConnectionError:
            raise ConnectionError("The Speaker Recognition service is "
                                  "unavailable.")
        try:
            return response.json()
        except:
            return response.text

