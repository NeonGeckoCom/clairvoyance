from functools import wraps
from flask import Flask, make_response
from flask import request, Response
import os
from os.path import join, exists
import json
import tempfile
from clairvoyance.db import JsonDatabase, AUTH_DB, SPEAKERS_MODEL
from clairvoyance.classifiers import GaussianSpeakerRecognizer


app = Flask(__name__)


def nice_json(arg):
    response = make_response(json.dumps(arg, sort_keys = True, indent=4))
    response.headers['Content-type'] = "application/json"
    return response


def check_auth(api_key):
    """This function is called to check if a username /
    password combination is valid.
    """
    with JsonDatabase("users", AUTH_DB) as db:
        users = db.search_by_value("key", api_key)
        if len(users):
            return True
    return False


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to login with proper credentials', 401,
        {'WWW-Authenticate': 'Basic realm="Api Key Required"'})


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        if not auth or not check_auth(auth):
            return authenticate()
        return f(*args, **kwargs)

    return decorated


@app.route("/", methods=['GET'])
def hello():
    return nice_json({
        "uri": "/",
        "welcome to xerlok microservices": {}
    })


@app.route("/speaker_recognition/recognize", methods=['PUT'])
@requires_auth
def id_user():
    auth = request.headers.get('Authorization', '')
    wav_data = request.data
    wav_file = join(tempfile.gettempdir(), auth + "rec.wav")
    with open(wav_file, "wb") as f:
        f.write(wav_data)

    model = join(SPEAKERS_MODEL, auth + ".pkl")

    if not model or not exists(model):
        res = {"error": "no trained model available"}
    else:
        recognizer = GaussianSpeakerRecognizer(model=model)
        res = recognizer.recognize(wav_file)
    try:
        os.remove(wav_file)
    except Exception as e:
        pass
    return nice_json(res)


@app.route("/speaker_recognition/train/<user>", methods=['PUT'])
@requires_auth
def train_user(user):
    auth = request.headers.get('Authorization', '')
    wav_data = request.data
    wav_file = join(tempfile.gettempdir(), auth + "train.wav")
    with open(wav_file, "wb") as f:
        f.write(wav_data)

    model = join(SPEAKERS_MODEL, auth + ".pkl")

    recognizer = GaussianSpeakerRecognizer(model=model)
    res = recognizer.train_single_file(user, wav_file)

    try:
        os.remove(wav_file)
    except Exception as e:
        pass

    return nice_json(res)


if __name__ == "__main__":

    port = 5678
    app.run(port=port, debug=True)
