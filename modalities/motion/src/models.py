import os
import torch
import torch.nn as nn

class MotionLSTM(nn.Module):
    """
    Sequence classifier (LSTM) for human action tracking.
    Expected Input: 99 features (33 MediaPipe Pose keypoints * 3 coordinates)
    """
    def __init__(self, input_size=99, hidden_size=128, num_layers=3, num_classes=9, dropout=0.4):
        super(MotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, -1)  # shape -> (batch, seq_len, 99)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # Get final hidden state
        out = self.fc(last_hidden)
        return out
