import torch.nn as nn

# Define the ChatbotModel class inheriting from nn.Module
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ChatbotModel, self).__init__()
        # Fully connected layer from input to hidden size
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation function
        self.relu = nn.ReLU()
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.3)
        # Fully connected layer from hidden to output size
        self.fc2 = nn.Linear(hidden_size, output_size)
        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Pass input through the first layer and activation
        x = self.fc1(x)
        x = self.relu(x)
        # Apply dropout
        x = self.dropout(x)
        # Pass through the second layer and apply softmax
        x = self.fc2(x)
        return self.softmax(x)
