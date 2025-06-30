import json
import numpy as np
from typing import List

class TrainingDataLoader:
    """
    Loads and prepares training data for neural network training.
    """
    
    def __init__(self, data_path: str = "training_data.json"):
        """
        Initialize training data loader.
        
        Args:
            data_path: Path to training data file
        """
        self.data_path = data_path
        self.data = None
    
    def load_data(self) -> List[dict]:
        """Load training data from file."""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        return self.data
    
    def prepare_training_batch(self, batch_size: int = 32, shuffle: bool = True) -> tuple:
        """
        Prepare training batches for neural network.
        
        Args:
            batch_size: Size of training batches
            shuffle: Whether to shuffle the data
            
        Returns:
            tuple: (fens, policy_targets, value_targets) for training
        """
        if self.data is None:
            self.load_data()
        
        data = self.data.copy()
        if shuffle:
            import random
            random.shuffle(data)
        
        # Prepare batches
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            
            fens = [ex['fen'] for ex in batch_data]
            policy_targets = np.array([ex['search_probabilities'] for ex in batch_data], dtype=np.float32)
            value_targets = np.array([ex['winner'] for ex in batch_data], dtype=np.float32)
            
            batches.append((fens, policy_targets, value_targets))
        
        return batches
    
    def get_data_stats(self) -> dict:
        """Get statistics about the loaded data."""
        if self.data is None:
            self.load_data()
        
        winners = [ex['winner'] for ex in self.data]
        return {
            "total_positions": len(self.data),
            "white_perspective_wins": sum(1 for w in winners if w > 0),
            "black_perspective_wins": sum(1 for w in winners if w < 0),
            "draws": sum(1 for w in winners if w == 0),
            "win_rate": (sum(1 for w in winners if w != 0) / len(winners)) * 100
        }