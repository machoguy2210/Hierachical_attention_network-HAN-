from model import HierarchicalAttentionNetwork as HAN
import torch
from utils import read_vocab, load_glove
from data_reader import DataReader

if __name__ == '__main__':
    embedding_dim = 200
    hidden_dim = 50
    num_classes = 2

    # Load the vocabulary
    vocab = read_vocab('sample_data-w2i.pkl')
    word_embedding = load_glove('glove.6B/glove.6B.200d.txt', 200, vocab)
    # Example usage
    vocab_size = len(word_embedding)  # Replace with the actual vocab size
    
    # Initialize the HAN model
    HAN_model = HAN(embedding_dim, hidden_dim)

    #Load the train/test data
    data_reader = DataReader('sample_data_train.pkl', 'sample_data_test.pkl')

    optimizer = torch.optim.Adam(HAN_model.parameters(), lr=0.001)
    criterion = torch.nn.NLLLoss()

    num_epochs = 10

    for epoch in range(num_epochs):
        
    
    
    