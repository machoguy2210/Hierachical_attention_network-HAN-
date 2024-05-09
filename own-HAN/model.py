import torch
import torch.nn as nn
import torch.nn.functional as F


class WordLevel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
        super(WordLevel, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, bidirectional=True)
        self.input_size = embedding_dim
        self.hidden_size = hidden_dim * 2
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.U = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        # input: (batch_size, max_seq_len)
        output, _ = self.gru(input)  # (batch_size, max_seq_len, 2*hidden_dim)
        # Áp dụng phép biến đổi tuyến tính W cho glove vectors
        transformed_vectors = self.W(output)  # Shape: (seq_len, hidden_size)

        # Tính điểm attention
        attention_scores = self.U(torch.tanh(transformed_vectors))  # Shape: (seq_len, 1)
        attention_scores = F.softmax(attention_scores, dim=0)  # Áp dụng softmax

        # Tính tổng có trọng số của glove vectors
        sentence_representation = torch.sum(attention_scores * output, dim=0)  # Shape: (hidden_size,)

        return sentence_representation

class SentenceLevel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers=1):
        super(SentenceLevel, self).__init__()
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, bidirectional=True)
        self.input_size = embedding_dim
        self.hidden_size = hidden_dim * 2
        self.W = nn.Linear(self.hidden_size, self.hidden_size)
        self.U = nn.Linear(self.hidden_size, 1)

    def forward(self, input):
        # input: (batch_size, max_seq_len)
        output, _ = self.gru(input)  # (batch_size, max_seq_len, 2*hidden_dim)
        # Áp dụng phép biến đổi tuyến tính W cho glove vectors
        transformed_vectors = self.W(output)  # Shape: (seq_len, hidden_size)

        # Tính điểm attention
        attention_scores = self.U(torch.tanh(transformed_vectors))  # Shape: (seq_len, 1)
        attention_scores = F.softmax(attention_scores, dim=0)  # Áp dụng softmax

        # Tính tổng có trọng số của glove vectors
        sentence_representation = torch.sum(attention_scores * output, dim=0)  # Shape: (hidden_size,)

        return sentence_representation

class HierarchicalAttentionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_gru_dim):
        super(HierarchicalAttentionNetwork, self).__init__()
        self.word_level = WordLevel(embedding_dim, hidden_gru_dim)
        self.sentence_level = SentenceLevel(hidden_gru_dim*2, hidden_gru_dim)
        self.classifier = nn.Linear(hidden_gru_dim*2, 1)

    def forward(self,input):
        s = []
        for x in input:
            print(x)
            s.append(self.word_level(x))
        s = torch.stack(s, dim=0)
        v = self.sentence_level(s)
        return self.classifier(v)
        

if __name__ == '__main__':
    '''
    # Load the vocabulary
    vocab = read_vocab('sample_data-w2i.pkl')
    word_embedding = load_glove('glove.6B/glove.6B.50d.txt', 50, vocab) 
    # Example usage
    vocab_size = len(word_embedding)  # Replace with the actual vocab size
    embedding_dim = 50
    hidden_dim = 50
    num_layers = 2

    # Create a sample input (batch_size=1, max_seq_len=10)
    sample_input = torch.tensor([[2, 3, 4]])  # Replace with actual word indices

    embedding = nn.Embedding(vocab_size, embedding_dim)
    embedding.weight.data.copy_(torch.tensor(word_embedding))

    sample_input = embedding(sample_input)
    print(sample_input)

    '''
    embedding_dim = 10
    hidden_dim = 5
    num_layers = 1

    # Initialize the WordEncoder
    HAN_model = HierarchicalAttentionNetwork(embedding_dim, hidden_dim)

    word_encoder = WordLevel(embedding_dim, hidden_dim, num_layers)

    # Create a sample input
    sample_input = torch.randn(5, 3, 10)  # (batch_size, max_seq_len, embedding_dim)
    
    
    v = HAN_model(sample_input)
    print(v)

    


