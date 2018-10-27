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
