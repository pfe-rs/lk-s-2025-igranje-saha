import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess
import chess.engine
from typing import List, Tuple, Optional
import struct
import os
from numba import jit, njit
import threading
from concurrent.futures import ThreadPoolExecutor

class ClippedReLU(nn.Module):
    """Custom Clipped ReLU activation: clamp(x, 0, 1)"""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)

class NNUEFeatures:
    """
    NNUE Feature representation using HalfKP (King-Piece) features
    """
    
    # Feature dimensions
    PIECE_TYPES = 5  # P, N, B, R, Q (excluding Kings)
    COLORS = 2       # White, Black
    SQUARES = 64     # 8x8 board
    
    # HalfKP feature size: King position (64) * Piece types without kings (10) * Square (64)
    # 10 piece types = 5 piece types * 2 colors (excluding both kings)
    FEATURE_SIZE = SQUARES * (PIECE_TYPES * COLORS) * SQUARES  # 64 * 10 * 64 = 40960
    
    @staticmethod
    @njit
    def piece_to_index(piece_type: int, color: int) -> int:
        """Convert piece type and color to feature index (excluding kings)"""
        if piece_type == 6:  # King
            return -1  # Kings are not included in features
        return (piece_type - 1) + (0 if color else 5)  # 0-4 for white, 5-9 for black
    
    @staticmethod
    def get_halfkp_features(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract HalfKP features for both perspectives (white and black)
        Features: King square (64) × Non-king pieces (10) × Piece square (64)
        Returns: (white_features, black_features)
        """
        white_features = np.zeros(NNUEFeatures.FEATURE_SIZE, dtype=np.float32)
        black_features = np.zeros(NNUEFeatures.FEATURE_SIZE, dtype=np.float32)
        
        # Find king positions
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)
        
        if white_king_sq is None or black_king_sq is None:
            return white_features, black_features
        
        # Process each piece on the board (excluding kings)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue  # Skip empty squares and kings
                
            piece_idx = NNUEFeatures.piece_to_index(piece.piece_type, piece.color)
            
            # White perspective feature
            # Formula: king_square * 10 * 64 + piece_type * 64 + piece_square
            white_feature_idx = white_king_sq * 10 * 64 + piece_idx * 64 + square
            white_features[white_feature_idx] = 1.0
            
            # Black perspective feature (mirrored)
            black_square = chess.square_mirror(square)
            black_king_mirror = chess.square_mirror(black_king_sq)
            # For black perspective, flip the piece color in the encoding
            black_piece_idx = NNUEFeatures.piece_to_index(piece.piece_type, not piece.color)
            black_feature_idx = black_king_mirror * 10 * 64 + black_piece_idx * 64 + black_square
            black_features[black_feature_idx] = 1.0
            
        return white_features, black_features
    
# Pure PyTorch alternative (no NumPy conversion needed)
class PureTorchSIMDLinear(nn.Module):
    """Pure PyTorch implementation avoiding NumPy compatibility issues"""
    def __init__(self, input_size, output_size, bits=16):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.bits = bits
        self.scale_factor = (2 ** (bits - 1)) - 1
        
        # Float weights for training
        self.weight_float = nn.Parameter(torch.randn(output_size, input_size) * 0.1)
        self.bias_float = nn.Parameter(torch.zeros(output_size))
        
        # Quantized buffers
        if bits == 16:
            self.register_buffer('weight_quant', torch.zeros(output_size, input_size, dtype=torch.int16))
            self.register_buffer('bias_quant', torch.zeros(output_size, dtype=torch.int16))
        else:  # 8-bit
            self.register_buffer('weight_quant', torch.zeros(output_size, input_size, dtype=torch.int8))
            self.register_buffer('bias_quant', torch.zeros(output_size, dtype=torch.int8))
    
    def quantize_weights(self):
        """Quantize weights for inference"""
        max_val = self.scale_factor
        min_val = -max_val if self.bits == 16 else -max_val - 1
        
        self.weight_quant.data = torch.clamp(
            (self.weight_float.data * self.scale_factor).round(),
            min_val, max_val
        ).to(self.weight_quant.dtype)
        
        self.bias_quant.data = torch.clamp(
            (self.bias_float.data * self.scale_factor).round(),
            min_val, max_val
        ).to(self.bias_quant.dtype)
    
    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_float, self.bias_float)
        else:
            # Quantized inference using pure PyTorch
            x_quant = torch.clamp(
                (x * self.scale_factor).round(),
                0 if self.bits == 8 else -self.scale_factor,
                self.scale_factor
            ).to(self.weight_quant.dtype)
            
            # Use torch.matmul which automatically uses optimized BLAS
            result = torch.matmul(x_quant.float(), self.weight_quant.float().t())
            result += self.bias_quant.float()
            
            return result / self.scale_factor

class NNUENetwork(nn.Module):
    """Pure PyTorch NNUE avoiding NumPy compatibility issues"""
    def __init__(self, input_size=81920, first_hidden=512, second_hidden=32):
        super().__init__()
        
        # First layer - mirrored weights (16-bit)
        self.input_layer1 = PureTorchSIMDLinear(input_size // 2, first_hidden // 2, bits=16)
        self.input_layer2 = PureTorchSIMDLinear(input_size // 2, first_hidden // 2, bits=16)
        
        # Share weights between the two halves
        self.input_layer2.weight_float = self.input_layer1.weight_float
        
        self.relu1 = ClippedReLU()
        
        # Hidden layers (8-bit)
        self.hidden1 = PureTorchSIMDLinear(first_hidden, second_hidden, bits=8)
        self.relu2 = ClippedReLU()
        
        self.hidden2 = PureTorchSIMDLinear(second_hidden, second_hidden, bits=8)
        self.relu3 = ClippedReLU()
        
        # Output layer
        self.output_layer = PureTorchSIMDLinear(second_hidden, 1, bits=8)

        # Accumulator for efficient incremental updates
        self.register_buffer('accumulator', torch.zeros(1, first_hidden))
        self.accumulator_valid = False
    
    def quantize_all_weights(self):
        """Quantize all layers"""
        self.input_layer1.quantize_weights()
        self.input_layer2.quantize_weights()
        self.hidden1.quantize_weights()
        self.hidden2.quantize_weights()
        self.output_layer.quantize_weights()
    
    def reset_accumulator(self):
        """Reset the accumulator for incremental updates"""
        self.accumulator.zero_()
        self.accumulator_valid = False
    
    def incremental_update(self, added_features, removed_features):
        """
        Efficiently update the accumulator based on piece moves
        Only updates the changed features instead of recomputing everything
        """
        if not self.accumulator_valid:
            return False
        
        # Add new features
        for feature_idx in added_features:
            if feature_idx < self.input_layer.half_input:
                # First half
                self.accumulator[0, :self.input_layer.half_output] += \
                    self.input_layer.weight_float[:, feature_idx]
            else:
                # Second half
                mapped_idx = feature_idx - self.input_layer.half_input
                self.accumulator[0, self.input_layer.half_output:] += \
                    self.input_layer.weight_float[:, mapped_idx]
        
        # Remove old features
        for feature_idx in removed_features:
            if feature_idx < self.input_layer.half_input:
                # First half
                self.accumulator[0, :self.input_layer.half_output] -= \
                    self.input_layer.weight_float[:, feature_idx]
            else:
                # Second half
                mapped_idx = feature_idx - self.input_layer.half_input
                self.accumulator[0, self.input_layer.half_output:] -= \
                    self.input_layer.weight_float[:, mapped_idx]
        
        return True
    
    def forward_from_accumulator(self):
        """
        Forward pass using the pre-computed accumulator
        Much faster for position evaluation during search
        """
        if not self.accumulator_valid:
            raise ValueError("Accumulator not valid. Call forward() first or update incrementally.")
        
        x = self.relu1(self.accumulator + self.input_layer.bias_float)
        x = self.relu2(self.hidden1(x))
        x = self.relu3(self.hidden2(x))
        x = self.output_layer(x)
        
        return x
    
    def forward(self, x):
        # Split input
        x1 = x[:, :x.shape[1]//2]
        x2 = x[:, x.shape[1]//2:]
        
        # Process both halves
        out1 = self.input_layer1(x1)
        out2 = self.input_layer2(x2)
        
        # Combine and continue
        x = torch.cat([out1, out2], dim=1)
        x = self.relu1(x)
        x = self.relu2(self.hidden1(x))
        x = self.relu3(self.hidden2(x))
        x = self.output_layer(x)
        
        return x

class ChessDataset(Dataset):
    """
    Dataset class for loading chess positions and evaluations
    """
    
    def __init__(self, positions: List[str], evaluations: List[float]):
        self.positions = positions
        self.evaluations = evaluations
        
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        fen = self.positions[idx]
        evaluation = self.evaluations[idx]
        
        board = chess.Board(fen)
        white_features, black_features = NNUEFeatures.get_halfkp_features(board)
        
        return {
            'white_features': torch.tensor(white_features, dtype=torch.float32),
            'black_features': torch.tensor(black_features, dtype=torch.float32),
            'evaluation': torch.tensor(evaluation, dtype=torch.float32)
        }

class NNUETrainer:
    """
    Training class for NNUE network with optimizations
    """
    
    def __init__(self, 
                 model: NNUENetwork,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with weight decay for regularization
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            verbose=True
        )
        
        # Loss function - Mean Squared Error
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            white_features = batch['white_features'].to(self.device)
            black_features = batch['black_features'].to(self.device)
            targets = batch['evaluation'].to(self.device).unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(white_features, black_features)
            
            # Compute loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, dataloader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                white_features = batch['white_features'].to(self.device)
                black_features = batch['black_features'].to(self.device)
                targets = batch['evaluation'].to(self.device).unsqueeze(1)
                
                outputs = self.model(white_features, black_features)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              num_epochs: int = 100,
              save_path: str = 'nnue_model.pth'):
        """Full training loop with validation"""
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.6f}')
            print(f'  Val Loss: {val_loss:.6f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.8f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f'  New best model saved!')
            
            print('-' * 50)

class NNUEEngine:
    """
    Chess engine using NNUE for position evaluation
    """
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = NNUENetwork()
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)
        
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a chess position using NNUE"""
        white_features, black_features = NNUEFeatures.get_halfkp_features(board)
        
        with torch.no_grad():
            white_tensor = torch.tensor(white_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            black_tensor = torch.tensor(black_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            evaluation = self.model(white_tensor, black_tensor)
            
            # Convert to centipawns and flip for black to move
            score = evaluation.item() * 100
            if not board.turn:  # Black to move
                score = -score
                
        return score
    
    def search(self, board: chess.Board, depth: int = 4) -> Tuple[chess.Move, float]:
        """
        Simple minimax search with alpha-beta pruning
        """
        def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
            if depth == 0 or board.is_game_over():
                return self.evaluate_position(board)
            
            if maximizing:
                max_eval = float('-inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval_score = minimax(board, depth - 1, alpha, beta, False)
                    board.pop()
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                return max_eval
            else:
                min_eval = float('inf')
                for move in board.legal_moves:
                    board.push(move)
                    eval_score = minimax(board, depth - 1, alpha, beta, True)
                    board.pop()
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                return min_eval
        
        best_move = None
        best_score = float('-inf') if board.turn else float('inf')
        
        for move in board.legal_moves:
            board.push(move)
            score = minimax(board, depth - 1, float('-inf'), float('inf'), not board.turn)
            board.pop()
            
            if board.turn:  # White to move
                if score > best_score:
                    best_score = score
                    best_move = move
            else:  # Black to move
                if score < best_score:
                    best_score = score
                    best_move = move
        
        return best_move, best_score

# Utility functions for data loading and preprocessing
def load_training_data(file_path: str) -> Tuple[List[str], List[float]]:
    """
    Load training data from file
    Expected format: FEN evaluation
    """
    positions = []
    evaluations = []
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')  # Assuming tab-separated
            if len(parts) >= 2:
                fen = parts[0]
                try:
                    eval_score = float(parts[1])
                    positions.append(fen)
                    evaluations.append(eval_score)
                except ValueError:
                    continue
    
    return positions, evaluations

def create_data_loaders(positions: List[str], 
                       evaluations: List[float],
                       batch_size: int = 1024,
                       train_split: float = 0.8,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    # Split data
    total_size = len(positions)
    train_size = int(total_size * train_split)
    
    train_positions = positions[:train_size]
    train_evaluations = evaluations[:train_size]
    val_positions = positions[train_size:]
    val_evaluations = evaluations[train_size:]
    
    # Create datasets
    train_dataset = ChessDataset(train_positions, train_evaluations)
    val_dataset = ChessDataset(val_positions, val_evaluations)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

# Example usage and training script
def main():
    """Main training function"""
    
    # Configuration
    BATCH_SIZE = 2048
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    HIDDEN_SIZE = 256
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Load training data
    print("Loading training data...")
    positions, evaluations = load_training_data('training_data.txt')  # Replace with your file
    print(f"Loaded {len(positions)} positions")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        positions, evaluations, 
        batch_size=BATCH_SIZE
    )
    
    # Create model
    print("Creating NNUE model...")
    model = NNUENetwork(
        feature_size=NNUEFeatures.FEATURE_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = NNUETrainer(
        model=model,
        learning_rate=LEARNING_RATE,
        device=DEVICE
    )
    
    # Train model
    print("Starting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        save_path='nnue_chess_model.pth'
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main()