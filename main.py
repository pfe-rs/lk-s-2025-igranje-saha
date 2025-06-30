import torch
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from ChessAI import *
from TrainingDataCollector import *

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)

class DualHeadChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(20, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.res_blocks = nn.Sequential(*[ResBlock(256) for _ in range(8)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 1024),  
            nn.ReLU(),
            nn.Dropout(0.5),  # Regularization
            nn.Linear(1024, 4672)  # <- 4672
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    def predict(self, fen):
        input = fen_to_tensor(sample_fen)

        input = input.unsqueeze(0).to(device)

        # Method 1: Direct call (most common)
        with torch.no_grad():  # Disable gradient computation for inference
            policy_output, value_output = model(input)

        return policy_output, value_output
    
def fen_to_tensor(fen):
        board_tensor = torch.zeros((20, 8, 8), dtype=torch.float32)
        board = chess.Board(fen)
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = 7 - square // 8, square % 8
                board_tensor[piece_map[piece.symbol()], rank, file] = 1
        board_tensor[12] = int(board.turn)
        board_tensor[13] = int(board.has_kingside_castling_rights(chess.WHITE))
        board_tensor[14] = int(board.has_queenside_castling_rights(chess.WHITE))
        board_tensor[15] = int(board.has_kingside_castling_rights(chess.BLACK))
        board_tensor[16] = int(board.has_queenside_castling_rights(chess.BLACK))
        board_tensor[17] = int(board.has_legal_en_passant())
        board_tensor[18] = board.halfmove_clock / 50.0
        board_tensor[19] = board.fullmove_number / 100.0
        return board_tensor
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DualHeadChessNet().to(device)

model.load_state_dict(torch.load('chess_dualhead_best.pth', map_location=device))
model.eval()  # Set to evaluation mode

# Getting output from the network
# Your input should be shape (batch_size, 20, 8, 8) for your chess board
sample_fen = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
policy_output, value_output = model.predict(sample_fen)
    
print(f"Policy output shape: {policy_output.shape}")  # Should be (1, 4672)
print(f"Value output shape: {value_output.shape}")    # Should be (1, 1)

print(f"Policy output value: {policy_output}")
print(f"Value output value: {value_output}")

train_collector = TrainingDataCollector()
chess_ai = ChessAI(model=model, c_puct=1.0, num_simulations=10, stochastic_selection=True, data_collector=train_collector, opening_book_file='chess_opening_book.json')

chess_ai.generate_training_data(num_games=1)