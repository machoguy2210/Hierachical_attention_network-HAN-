import nltk
import pandas as pd
import itertools
import pickle
import re

WORD_CUT_OFF = 1

def remove_punctuation(text):
  """Loại bỏ dấu câu khỏi văn bản sử dụng biểu thức chính quy."""
  return re.sub(r"[^\w\s]", "", text)

def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    print('{}, shape={}'.format(file_path, data.shape))
    return data

def build_vocab(docs,save_path):
    print('Building vocab ...')
    sents = itertools.chain(*[remove_punctuation(text).lower().split('.') for text in docs])
    tokenized_sents = [sent.split() for sent in sents]    
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
    print("%d unique words found" % len(word_freq.items()))
    retained_words = [w for (w, f) in word_freq.items() if f > WORD_CUT_OFF]
    print("%d words retained" % len(retained_words))
    word_to_index = {'PAD': 0, 'UNK': 1}
    for i, w in enumerate(retained_words):
        word_to_index[w] = i + 2
    index_to_word = {i: w for (w, i) in word_to_index.items()}
    print("Vocabulary size = %d" % len(word_to_index))
    print(word_to_index)
    print(index_to_word) 
    
    with open('{}-w2i.pkl'.format(save_path), 'wb') as f:
        pickle.dump(word_to_index, f)
    
    with open('{}-i2w.pkl'.format(save_path), 'wb') as f:
        pickle.dump(index_to_word, f)
    return word_to_index

def process_and_save(word_to_index, data, out_file):
    mapped_data = []
    for label, doc in zip(data[0],data[1]):
        mapped_doc = [[word_to_index.get(word, 1) for word in remove_punctuation(sent).lower().split()] for sent in doc.split('.') if sent != '']
        mapped_data.append((label, mapped_doc))

    with open(out_file, 'wb') as f:
        pickle.dump(mapped_data, f) 

if __name__ == '__main__': 
    train_data = read_data('train_data/train.csv')
    word_to_index =  build_vocab(train_data[1], 'train_data/vocab')
    process_and_save(word_to_index, train_data, 'train_data/train.pkl')
    