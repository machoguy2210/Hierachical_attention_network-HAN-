import nltk
import pandas as pd
import itertools
import pickle

WORD_CUT_OFF = 5

def build_vocab(docs, save_path):
  print('Building vocab ...')

  sents = itertools.chain(*[text.split('<sssss>') for text in docs])
  tokenized_sents = [sent.split() for sent in sents]

  # Count the word frequencies
  word_freq = nltk.FreqDist(itertools.chain(*tokenized_sents))
  print("%d unique words found" % len(word_freq.items()))

  # Cut-off
  retained_words = [w for (w, f) in word_freq.items() if f > WORD_CUT_OFF]
  print("%d words retained" % len(retained_words))

  # Get the most common words and build index_to_word and word_to_index vectors
  # Word index starts from 2, 1 is reserved for UNK, 0 is reserved for padding
  word_to_index = {'PAD': 0, 'UNK': 1}
  for i, w in enumerate(retained_words):
    word_to_index[w] = i + 2
  index_to_word = {i: w for (w, i) in word_to_index.items()}

  print("Vocabulary size = %d" % len(word_to_index))

  with open('{}-w2i.pkl'.format(save_path), 'wb') as f:
    pickle.dump(word_to_index, f)

  with open('{}-i2w.pkl'.format(save_path), 'wb') as f:
    pickle.dump(index_to_word, f)

  return word_to_index

def process_and_save(word_to_index, data, labels, out_file):
  mapped_data = []
  for label, doc in zip(labels, data):
    mapped_doc = [[word_to_index.get(word, 1) for word in sent.split()] for sent in doc.split('<sssss>')]
    mapped_data.append((label, mapped_doc))

  with open(out_file, 'wb') as f:
    pickle.dump(mapped_data, f)

if __name__ == '__main__': 
    with open('train_data/yelp_2013_texts.txt', 'r', encoding="utf-8") as f:
        data = f.readlines()
    with open('train_data/yelp_2013_score.txt', 'r') as f:
        labels = list(map(int,f.read().splitlines()))
    data = pd.DataFrame(data)
    word_to_index = build_vocab(data[0], 'train_data/vocab')
    process_and_save(word_to_index, data[0], labels, 'train_data/data.pkl')
    