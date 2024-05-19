import pickle
import numpy as np
import random

def read_vocab(vocab_file):
  print('Loading vocabulary ...')
  with open(vocab_file, 'rb') as f:
    word_to_index = pickle.load(f)
    print('Vocabulary size = %d' % len(word_to_index))
    return word_to_index

def load_glove(glove_file, emb_size, vocab):
    print('Loading Glove pre-trained word embeddings ...')
    embedding_weights = {}
    f = open(glove_file, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.array(values[1:], dtype='float32')
        embedding_weights[word] = vector
    f.close()
    print('Loaded %d word vectors.' % len(embedding_weights))

    embedding_matrix = np.random.uniform(-0.5, 0.5, (len(vocab), emb_size)) / emb_size

    oov_count = 0
    for word, i in vocab.items():
        embedding_vector = embedding_weights.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            oov_count += 1
    print('Number of OOV words: %d' % oov_count)
    return embedding_matrix

def read_data(file_path, num_classes=5):        
        print('Reading data from %s' % file_path)
        new_data = []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            random.shuffle(data)
            for label, doc in data:
                label -= 1
                assert label >= 0 and label < num_classes

                new_data.append((doc, label))

        return new_data