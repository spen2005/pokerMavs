class Dataset:
    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def __iter__(self):
        return iter(self.data)

    def clear(self):
        self.data.clear()