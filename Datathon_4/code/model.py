
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class LSTMNet(nn.Module):
      def __init__(self, input_size, hidden_size, num_classes):
          super(LSTMNet, self).__init__()
          self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
          self.fc = nn.Linear(hidden_size, num_classes)

      def forward(self, x):
          h, _ = self.lstm(x)
          out = self.fc(h[:, -1, :])
          return out

class EnhancedLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super(EnhancedLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h, _ = self.lstm(x)
        out = self.fc(h[:, -1, :])
        return out

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super(RNNNet, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=2, batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        out = self.fc(h[:, -1, :])
        return out