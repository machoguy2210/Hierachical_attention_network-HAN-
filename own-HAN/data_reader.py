import pickle
import numpy as np

class DataReader:
    def __init__(self, train_file, test_file, max_word_length=50, max_sent_length=20, num_classes=2):
        self.max_word_length = max_word_length
        self.max_sent_length = max_sent_length
        self.num_classes = num_classes

        self.train_data = self._read_data(train_file)
        #self.test_data = self._read_data(test_file)

    def _read_data(self, file_path):
        print('Reading data from %s' % file_path)
        neww_data = []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            for label, doc in data:
                doc = doc[:self.max_sent_length]
                doc = [sent[:self.max_word_length] for sent in doc]

                label -= 1
                assert label >= 0 and label < self.num_classes

                neww_data.append((doc, label))

        return neww_data
    