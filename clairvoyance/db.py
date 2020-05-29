from os import makedirs
from os.path import join, exists, expanduser
from json_database import JsonDatabase

_db_path = join(expanduser("~"), ".clairvoyance")
AUTH_DB = join(_db_path, "auth.db")
SPEAKERS_MODEL = join(_db_path, "speakers")
if not exists(_db_path):
    makedirs(_db_path)
if not exists(SPEAKERS_MODEL):
    makedirs(SPEAKERS_MODEL)
