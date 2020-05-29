from os.path import exists
from clairvoyance.classifiers import gaussian
from clairvoyance.utils import read_wav


class GaussianSpeakerRecognizer:
    def __init__(self, model):
        self.model = gaussian.ModelInterface()
        self.model_path = model
        self.load_model()

    def load_model(self):
        if exists(self.model_path):
            self.model = self.model.load(self.model_path)

    def train_single_file(self, user="unknown", wav_file=None):
        if user == "unknown" or wav_file is None:
            return {"success": False, "error": "invalid args"}
        try:
            wav_file = wav_file
            fs, signal = read_wav(wav_file)
            self.model.enroll(user, fs, signal)
            for idx in self.model.gmm_map:
                if self.model.gmm_map[idx] == user:
                    self.model.retrain_user(user)
            else:
                self.model.train()
            self.model.dump(self.model_path)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def recognize(self, wav_file):
        fs, signal = read_wav(wav_file)
        label, score = self.model.predict(fs, signal)
        score = 1 + score
        if score < 0:
            score = 0.0011111111
        if score > 1:
            score = 0.9999999899
        return {label: score}
