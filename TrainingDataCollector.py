import chess
import json
from utils import *

class TrainingDataCollector:
    """
    Collects and manages training data from self-play games.
    """
    
    def __init__(self, save_path: str = "training_data.json"):
        """
        Initialize training data collector.
        
        Args:
            save_path: Path to save training data
        """
        self.save_path = save_path
        self.training_examples = []
    
    def add_game_data(self, game_result: dict) -> None:
        """
        Add training data from a completed game.
        
        Args:
            game_result: Game result dictionary from ChessAI.play_game()
        """
        game_winner = game_result['result']  # 1 for white win, -1 for black win, 0 for draw
        
        for i, move_data in enumerate(game_result['move_probabilities']):
            fen = move_data['fen']
            move_probs = move_data['move_probs']
            
            # Determine winner from current player's perspective
            board = chess.Board(fen)
            current_player_is_white = board.turn
            
            if game_winner == 0:
                winner_from_perspective = 0  # Draw
            elif game_winner == 1:  # White won
                winner_from_perspective = 1 if current_player_is_white else -1
            else:  # Black won (game_winner == -1)
                winner_from_perspective = -1 if current_player_is_white else 1
            
            # Convert move probabilities to 4672-length array
            legal_moves = set(board.legal_moves)
            search_probs_array = moves_dict_to_array(move_probs, legal_moves)
            
            training_example = {
                'fen': fen,
                'search_probabilities': search_probs_array.tolist(),  # Convert to list for JSON
                'winner': winner_from_perspective
            }
            
            self.training_examples.append(training_example)
    
    def save_data(self, append: bool = True) -> None:
        """
        Save training data to file.
        
        Args:
            append: If True, append to existing file. If False, overwrite.
        """
        if append:
            try:
                with open(self.save_path, 'r') as f:
                    existing_data = json.load(f)
                existing_data.extend(self.training_examples)
                data_to_save = existing_data
            except (FileNotFoundError, json.JSONDecodeError):
                data_to_save = self.training_examples
        else:
            data_to_save = self.training_examples
        
        with open(self.save_path, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"Saved {len(self.training_examples)} training examples to {self.save_path}")
        print(f"Total examples in file: {len(data_to_save)}")
    
    def clear_buffer(self) -> None:
        """Clear the current buffer of training examples."""
        self.training_examples = []
    
    def get_stats(self) -> dict:
        """Get statistics about collected data."""
        if not self.training_examples:
            return {"total": 0}
        
        winners = [ex['winner'] for ex in self.training_examples]
        return {
            "total": len(self.training_examples),
            "white_wins": sum(1 for w in winners if w > 0),
            "black_wins": sum(1 for w in winners if w < 0),
            "draws": sum(1 for w in winners if w == 0)
        }