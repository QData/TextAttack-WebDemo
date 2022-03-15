import pickle

from config import PRECOMPUTED_RESULTS_DICT_NAME, HISTORY

class Cache():
    def __init__(self, log=False):
        self.log = log
        self.cache = self.load_precomputed_results()

    def load_precomputed_results(self):
        try:
            precomputed_results = pickle.load(open(PRECOMPUTED_RESULTS_DICT_NAME, "rb"))
        except FileNotFoundError:
            precomputed_results = {}
        if self.log: print(f'Found {len(precomputed_results)} keys in pre-computed results.')
        return precomputed_results

    def add(self, key, data):
        self.cache = self.load_precomputed_results()
        self.cache[key] = data

        # update history
        if isinstance(data, list):
            self.cache[HISTORY] = data + self.cache.get(HISTORY, [])
        else:
            self.cache[HISTORY] = [data] + self.cache.get(HISTORY, [])

        pickle.dump(self.cache, open(PRECOMPUTED_RESULTS_DICT_NAME, 'wb'))
        if self.log: print(f'Successfully added {key} to the cache')

    def to_top(self, key):
        self.cache = self.load_precomputed_results()
        data, history = self.cache.get(key, None), self.cache.get(HISTORY, None)
        if not data or not history:
            return []

        if isinstance(data, list):
            for d in data:
                history.pop(history.index(d))
                history.insert(0, d)
        else:
            history.pop(history.index(data))
            history.insert(0, data)

    def exists(self, key):
        self.cache = self.load_precomputed_results()
        return key in self.cache

    def purge(self, key=None):
        self.cache = self.load_precomputed_results()
        if not key:
            self.cache.clear()
        elif key in self.cache:
            del self.cache[key]
        if self.log: print(f'Cache successfully purged')
        pickle.dump(self.cache, open(PRECOMPUTED_RESULTS_DICT_NAME, 'wb'))

    def get(self, key):
        self.cache = self.load_precomputed_results()
        return self.cache.get(key, [])