import os

import _pickle as pickle


class MyPickle:

    @staticmethod
    def write(x, path):
        with open(path, "wb") as f1:
            pickle.dump(x, f1)

    @staticmethod
    def read(path):
        if os.path.exists(path):
            with open(path, "rb") as f1:
                x = pickle.load(f1)
        else:
            x = None
        return x
