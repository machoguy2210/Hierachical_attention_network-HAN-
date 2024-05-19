from model import HierarchicalAttentionNetwork as HAN
import torch
import torch.nn as nn
import torch.optim as optim
from utils import read_vocab, load_glove, read_data
import pickle 
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    embedding_dim = 200
    hidden_dim = 50
    num_classes = 5

    # Load the vocabulary
    vocab = read_vocab('train_data/vocab-w2i.pkl')
    word_embedding = load_glove('glove.6B/glove.6B.200d.txt', 200, vocab)



    # Initialize the HAN model
    HAN_model = HAN(word_embedding, embedding_dim, hidden_dim)

    #Load the train/test data
    data = read_data('train_data/data.pkl', num_classes=num_classes)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    print('Train data size: {}'.format(len(train_data)))
    print('Test data size: {}'.format(len(test_data)))

    optimizer = optim.SGD(HAN_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1
    for epoch in range(num_epochs):
        for inputs,target in (train_data):
            optimizer.zero_grad()
            output = HAN_model(inputs).unsqueeze(0)
            target = torch.tensor([target])
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Test the model
    HAN_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for target, input in (test_data):
            output = HAN_model(input).unsqueeze(0)
            target = torch.tensor([target])
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        

    print('Accuracy: {}'.format(correct/total))



    

    
            
    
    
    