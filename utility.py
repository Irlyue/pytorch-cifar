import os
import config


class HistoryHandler:
    def __init__(self, filepath):
        self.filepath = filepath

    def __enter__(self):
        self.file = open(self.filepath, 'a+')
        self.file.__enter__()
        return self

    def __exit__(self, *args):
        self.file.__exit__(*args)

    def __call__(self, data):
        self.file.write(str(data) + '\n')


def load_config_from_environ():
    default = config.CONFIG.copy()
    for key, value in default.items():
        default[key] = type(value)(os.environ[key]) if key in os.environ else value
    return default
