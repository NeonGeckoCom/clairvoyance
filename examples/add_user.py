from clairvoyance.db import JsonDatabase, AUTH_DB
from jarbas_utils.security import random_key


def add_key(user="demo_user"):
    k = random_key(32)

    with JsonDatabase("users", AUTH_DB) as db:
        db.add_item({"user": user, "key": k})

    return k


# TODO argparse