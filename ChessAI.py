from MCTS import *
from TrainingDataCollector import *
from TrainingDataLoader import *
from OpeningBook import *

MAX_MOVES = 20

class ChessAI:
    """
    Complete Chess AI using MCTS with neural network.
    """
    
    def __init__(self, model, c_puct: float = 1.0, num_simulations: int = 800, 
                 stochastic_selection: bool = False, data_collector: TrainingDataCollector = None, opening_book_file: str = None):
        """
        Initialize Chess AI.
        
        Args:
            model: Trained neural network for position evaluation and move probabilities
            c_puct: Exploration parameter for MCTS
            num_simulations: Number of MCTS simulations per move
            stochastic_selection: Whether to use stochastic selection in MCTS
            data_collector: TrainingDataCollector instance for saving self-play data
        """
        self.mcts = MCTS(model, c_puct, stochastic_selection)
        self.num_simulations = num_simulations
        self.data_collector = data_collector

        self.opening_book = OpeningBook()
        
        if opening_book_file and os.path.exists(opening_book_file):
            self.opening_book.load_book(opening_book_file)
        else:
            print("No opening book loaded. Build one first!")
    
    def get_best_move(self, fen: str, move_number: int) -> chess.Move:
        """Get the best move using opening book or neural network."""
        # Try opening book first
        board = chess.Board(fen)
        book_move = self.opening_book.get_book_move(fen, move_number)
        if book_move:
            print(f"Opening book move: {book_move}")
            return board.parse_san(book_move)
        
        # Fall back to minimax
        print("Using neural network algorithm")
        root_node = create_root_node(board)
        best_edge = self.mcts.search(root_node, self.num_simulations)
        return best_edge.move
    
    def get_move_with_analysis(self, board: chess.Board, temperature: float = 0.0) -> tuple:
        """
        Get move with detailed analysis.
        
        Args:
            board: Current chess position
            temperature: Temperature for move selection (0 = deterministic)
            
        Returns:
            tuple: (best_move, move_probabilities, search_statistics)
        """
        root_node = create_root_node(board)
        best_edge = self.mcts.search(root_node, self.num_simulations)
        
        move_probs = self.mcts.get_move_probabilities(root_node, temperature)
        stats = self.mcts.get_search_statistics(root_node)
        
        return best_edge.move, move_probs, stats
    
    def play_game(self, opponent_ai=None, max_moves: int = MAX_MOVES, save_data: bool = True) -> dict:
        """
        Play a complete game and optionally save training data.
        
        Args:
            opponent_ai: Another ChessAI instance to play against (None for self-play)
            max_moves: Maximum number of moves before declaring draw
            save_data: Whether to save training data from this game
            
        Returns:
            dict: Game result with moves, outcome, etc.
        """
        board = chess.Board()
        moves = []
        move_probs_history = []
        
        if opponent_ai is None:
            opponent_ai = self  # Self-play
        
        move_count = 1
        while not board.is_game_over() and move_count <= max_moves:
            print(move_count)
            # Choose which AI makes the move
            current_ai = self if board.turn == chess.WHITE else opponent_ai
            
            # Get move with probabilities for training data
            move, move_probs, _ = current_ai.get_move_with_analysis(board, temperature=1.0)
            move = current_ai.get_best_move(board.fen(), (move_count + 1) // 2)
            
            # Record move and probabilities
            moves.append(move)
            move_probs_history.append({
                'fen': board.fen(),
                'move_probs': move_probs,
                'move_played': move
            })
            
            # Make the move
            board.push(move)
            move_count += 1
        
        # Determine result
        result = None
        if board.is_checkmate():
            result = 1 if not board.turn else -1  # Winner is the player who just moved
        elif board.is_stalemate() or board.is_insufficient_material():
            result = 0  # Draw
        else:
            result = 0  # Draw by move limit
        
        game_result = {
            'moves': moves,
            'result': result,
            'final_fen': board.fen(),
            'move_count': move_count,
            'move_probabilities': move_probs_history,
            'pgn': str(board)
        }
        
        # Save training data if requested and collector is available
        if save_data and self.data_collector is not None:
            self.data_collector.add_game_data(game_result)
        
        return game_result
    
    def generate_training_data(self, num_games: int, save_every: int = 10) -> None:
        """
        Generate training data through self-play.
        
        Args:
            num_games: Number of self-play games to generate
            save_every: Save data to file every N games
        """
        if self.data_collector is None:
            raise ValueError("No data collector provided. Initialize ChessAI with a TrainingDataCollector.")
        
        print(f"Starting generation of {num_games} self-play games...")
        
        for game_num in range(num_games):
            print(f"Playing game {game_num + 1}/{num_games}")
            
            # Play self-play game
            game_result = self.play_game(save_data=True)
            
            print(f"Game {game_num + 1} completed: "
                  f"Result = {game_result['result']}, "
                  f"Moves = {game_result['move_count']}")
            
            # Save periodically
            if (game_num + 1) % save_every == 0:
                self.data_collector.save_data(append=True)
                self.data_collector.clear_buffer()
                
                # Print stats
                loader = TrainingDataLoader(self.data_collector.save_path)
                stats = loader.get_data_stats()
                print(f"Training data stats after {game_num + 1} games: {stats}")
        
        # Save any remaining data
        if self.data_collector.training_examples:
            self.data_collector.save_data(append=True)
            self.data_collector.clear_buffer()
        
        print(f"Completed generation of {num_games} games!")
        
        # Final stats
        loader = TrainingDataLoader(self.data_collector.save_path)
        final_stats = loader.get_data_stats()
        print(f"Final training data stats: {final_stats}")
