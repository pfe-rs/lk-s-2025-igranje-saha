import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import chess
from typing import List, Tuple, Optional
import os

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
    FEATURE_SIZE = SQUARES * (PIECE_TYPES * COLORS) * SQUARES  # 64 * 10 * 64 = 40960
    
    @staticmethod
    def mirror_square(square: int) -> int:
        """Mirror square vertically (a1 -> a8, etc.)"""
        return chess.square(chess.square_file(square), 7 - chess.square_rank(square))
    
    @staticmethod
    def piece_to_index(piece_type: int, color: int) -> int:
        """Convert piece type and color to feature index (excluding kings)"""
        if piece_type == chess.KING:  # King (value 6)
            return -1
        return (piece_type - 1) + (0 if color else 5)  # 0-4 for white, 5-9 for black
    
    @staticmethod
    def get_perspective_features(board: chess.Board) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from perspective of side to move
        Returns: (stm_features, opponent_features)
        """
        stm = board.turn
        stm_king = board.king(stm)
        opponent_king = board.king(not stm)
        
        stm_features = np.zeros(NNUEFeatures.FEATURE_SIZE, dtype=np.float32)
        opponent_features = np.zeros(NNUEFeatures.FEATURE_SIZE, dtype=np.float32)
        
        if stm_king is None or opponent_king is None:
            return stm_features, opponent_features
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue
                
            # STM perspective feature
            stm_piece_idx = NNUEFeatures.piece_to_index(piece.piece_type, piece.color == stm)
            stm_feature_idx = stm_king * 10 * 64 + stm_piece_idx * 64 + square
            stm_features[stm_feature_idx] = 1.0
            
            # Opponent perspective feature (mirrored)
            mirrored_square = NNUEFeatures.mirror_square(square)
            opponent_piece_idx = NNUEFeatures.piece_to_index(piece.piece_type, piece.color != stm)
            opponent_king_mirror = NNUEFeatures.mirror_square(opponent_king)
            opponent_feature_idx = opponent_king_mirror * 10 * 64 + opponent_piece_idx * 64 + mirrored_square
            opponent_features[opponent_feature_idx] = 1.0
            
        return stm_features, opponent_features

class SIMDLinear(nn.Module):
    """Hidden layer implementation"""
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
    """NNUE implementation"""
    def __init__(self, feature_size=40960, first_hidden=256, second_hidden=32):
        super().__init__()
        self.feature_size = feature_size
        
        # Input layers - shared weights with mirroring relationship
        self.input_layer1 = SIMDLinear(feature_size, first_hidden, bits=16)
        self.input_layer2 = SIMDLinear(feature_size, first_hidden, bits=16)
        
        # Share weights between the two halves
        self.input_layer2.weight_float = self.input_layer1.weight_float
        self.input_layer2.bias_float = self.input_layer1.bias_float
        
        self.relu1 = ClippedReLU()
        
        # Hidden layers (8-bit)
        self.hidden1 = SIMDLinear(first_hidden, second_hidden, bits=8)
        self.relu2 = ClippedReLU()
        
        self.hidden2 = SIMDLinear(second_hidden, second_hidden, bits=8)
        self.relu3 = ClippedReLU()
        
        # Output layer
        self.output_layer = SIMDLinear(second_hidden, 1, bits=8)

    def quantize_all_weights(self):
        """Quantize all layers"""
        self.input_layer1.quantize_weights()
        self.input_layer2.quantize_weights()
        self.hidden1.quantize_weights()
        self.hidden2.quantize_weights()
        self.output_layer.quantize_weights()
    
    def forward(self, stm_features, opponent_features):
        # Process both perspectives
        stm_out = self.input_layer1(stm_features)
        opponent_out = self.input_layer2(opponent_features)
        
        # Combine and continue
        x = torch.cat([stm_out, opponent_out], dim=1)
        x = self.relu1(x)
        x = self.relu2(self.hidden1(x))
        x = self.relu3(self.hidden2(x))
        x = self.output_layer(x)
        
        return x * 100  # Scale to centipawns

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
        stm_features, opponent_features = NNUEFeatures.get_perspective_features(board)
        
        # Combine features into single tensor
        features = np.concatenate([stm_features, opponent_features])
        
        # Adjust evaluation to perspective of side to move
        if board.turn == chess.BLACK:
            evaluation = -evaluation
            
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'evaluation': torch.tensor(evaluation, dtype=torch.float32)
        }

class NNUETrainer:
    """Training class for NNUE network"""
    
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
            features = batch['features'].to(self.device)
            targets = batch['evaluation'].to(self.device).unsqueeze(1)
            
            # Split features into STM and opponent perspectives
            half_size = features.size(1) // 2
            stm_features = features[:, :half_size]
            opponent_features = features[:, half_size:]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(stm_features, opponent_features)
            
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
                features = batch['features'].to(self.device)
                targets = batch['evaluation'].to(self.device).unsqueeze(1)
                
                half_size = features.size(1) // 2
                stm_features = features[:, :half_size]
                opponent_features = features[:, half_size:]
                
                outputs = self.model(stm_features, opponent_features)
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
    """Chess engine using NNUE for position evaluation"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device
        self.model = NNUENetwork()
        self.model.to(device)
        
        # Load trained model
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.quantize_all_weights()
    
    def evaluate_position(self, board: chess.Board) -> float:
        """Evaluate a chess position using NNUE"""
        stm_features, opponent_features = NNUEFeatures.get_perspective_features(board)
        
        with torch.no_grad():
            stm_tensor = torch.tensor(stm_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            opponent_tensor = torch.tensor(opponent_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            evaluation = self.model(stm_tensor, opponent_tensor)
            
        return evaluation.item()
    
    def search(self, board: chess.Board, depth: int = 4) -> Tuple[chess.Move, float]:
        """Simple minimax search with alpha-beta pruning"""
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
    """Load training data from file
       Expected format: FEN evaluation"""
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
        first_hidden=HIDDEN_SIZE,
        second_hidden=32
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