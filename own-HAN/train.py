from model import HierarchicalAttentionNetwork as HAN
import torch
import torch.nn as nn
import torch.optim as optim
from utils import read_vocab, load_glove
import pickle 

if __name__ == '__main__':
    embedding_dim = 200
    hidden_dim = 50
    num_classes = 2

    # Load the vocabulary
    vocab = read_vocab('train_data/vocab-w2i.pkl')
    word_embedding = load_glove('glove.6B/glove.6B.200d.txt', 200, vocab)

    # Example usage
    vocab_size = len(word_embedding)  # Replace with the actual vocab size
    
    # Initialize the HAN model
    HAN_model = HAN(word_embedding, embedding_dim, hidden_dim)

    #Load the train/test data
    with open('train_data/train.pkl', 'rb') as f:
        train_data = pickle.load(f)

    optimizer = optim.SGD(HAN_model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 1

    for epoch in range(num_epochs):
        for target, input in (train_data):
            optimizer.zero_grad()
            output = HAN_model(input)
            target = torch.tensor([target-1]).float()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    
    
    