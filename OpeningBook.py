import json
import chess
import chess.pgn
import random
import zstandard as zstd
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import os
import sys

class OpeningBook:
    def __init__(self):
        """
        Initialize the opening book for processing Lichess database files.
        """
        self.book = {}
        self.max_opening_moves = 10
        self.min_games_threshold = 100  # Minimum games to include a move
        self.position_stats = defaultdict(lambda: defaultdict(int))
        
    def build_from_lichess_pgn(self, pgn_file_path: str, max_games: int = None, 
                              min_rating: int = 1800, time_controls: List[str] = None):
        """
        Build opening book from Lichess PGN database file.
        
        Args:
            pgn_file_path: Path to the Lichess PGN file (can be .pgn, .pgn.gz, or .pgn.bz2)
            max_games: Maximum number of games to process (None for all)
            min_rating: Minimum average rating of players
            time_controls: List of time controls to include (e.g., ['blitz', 'rapid', 'classical'])
        """
        print(f"Building opening book from: {os.path.basename(pgn_file_path)}")
        print(f"Min rating: {min_rating}")
        print(f"Time controls: {time_controls or 'All'}")
        
        # Reset stats
        self.position_stats = defaultdict(lambda: defaultdict(int))
        
        games_processed = 0
        games_used = 0
        
        try:
            # Open file (handle compression)
            pgn_file = self._open_pgn_file(pgn_file_path)
            
            while True:
                if max_games and games_processed >= max_games:
                    break
                
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                games_processed += 1
                
                # Progress indicator
                if games_processed % 10000 == 0:
                    print(f"Processed {games_processed:,} games, used {games_used:,}")
                
                # Filter games based on criteria
                if self._should_include_game(game, min_rating, time_controls):
                    self._process_game(game)
                    games_used += 1
            
            pgn_file.close()
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return
        
        print(f"\nTotal games processed: {games_processed:,}")
        print(f"Games used for book: {games_used:,}")
        
        # Convert statistics to opening book
        self._build_book_from_stats()
        
        print(f"Opening book created with {len(self.book):,} positions")
    
    def build_from_multiple_files(self, pgn_files: List[str], **kwargs):
        """
        Build opening book from multiple Lichess PGN files.
        
        Args:
            pgn_files: List of PGN file paths
            **kwargs: Arguments passed to build_from_lichess_pgn
        """
        print(f"Building opening book from {len(pgn_files)} files")
        
        # Reset stats
        self.position_stats = defaultdict(lambda: defaultdict(int))
        
        for i, pgn_file in enumerate(pgn_files):
            print(f"\n=== Processing file {i+1}/{len(pgn_files)}: {os.path.basename(pgn_file)} ===")
            
            # Process each file but don't rebuild book until the end
            temp_stats = self.position_stats.copy()
            self._process_single_file(pgn_file, **kwargs)
        
        # Build final book from combined statistics
        self._build_book_from_stats()
        print(f"\nFinal opening book: {len(self.book):,} positions")
    
    def _process_single_file(self, pgn_file_path: str, max_games: int = None,
                           min_rating: int = 1500, time_controls: List[str] = None):
        """Process a single PGN file and update statistics."""
        games_processed = 0
        games_used = 0
        
        try:
            pgn_file = self._open_pgn_file(pgn_file_path)
            
            while True:
                if max_games and games_processed >= max_games:
                    break
                
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                
                games_processed += 1
                
                if games_processed % 10000 == 0:
                    print(f"  {games_processed:,} games processed, {games_used:,} used")
                
                if self._should_include_game(game, min_rating, time_controls):
                    self._process_game(game)
                    games_used += 1
            
            pgn_file.close()
            print(f"  Finished: {games_processed:,} processed, {games_used:,} used")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    def _open_pgn_file(self, file_path: str):
        """Open PGN file, handling different compression formats."""
        if file_path.endswith('.zst'):
            return zstd.open(file_path, 'rt', encoding='utf-8')
        else:
            return open(file_path, 'r', encoding='utd-8')
    
    def _should_include_game(self, game, min_rating: int, time_controls: List[str]) -> bool:
        """
        Check if a game should be included based on filtering criteria.
        """
        try:
            headers = game.headers
            
            # Check time control
            if time_controls:
                time_control = headers.get('TimeControl', '')
                event = headers.get('Event', '').lower()
                
                # Map Lichess event names to time controls
                tc_found = False
                for tc in time_controls:
                    if tc.lower() in event:
                        tc_found = True
                        break
                if not tc_found:
                    return False
            
            # Check ratings
            white_elo = headers.get('WhiteElo', '?')
            black_elo = headers.get('BlackElo', '?')
            
            if white_elo == '?' or black_elo == '?':
                return False
            
            try:
                avg_rating = (int(white_elo) + int(black_elo)) / 2
                if avg_rating < min_rating:
                    return False
            except ValueError:
                return False
            
            # Check game length (avoid very short games)
            moves = list(game.mainline_moves())
            if len(moves) < 10:  # At least 5 moves per side
                return False
            
            # Check termination (avoid games that ended due to disconnection, etc.)
            termination = headers.get('Termination', '').lower()
            if 'abandon' in termination or 'unterminated' in termination:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _process_game(self, game):
        """
        Extract opening moves from a game and update statistics.
        """
        try:
            board = game.board()
            move_count = 0
            
            for move in game.mainline_moves():
                if move_count >= self.max_opening_moves:
                    break
                
                # Get current position
                fen = self._normalize_fen(board.fen())
                
                # Get move in standard algebraic notation
                san_move = board.san(move)
                
                # Update statistics
                self.position_stats[fen][san_move] += 1
                
                # Make the move
                board.push(move)
                move_count += 1
                
        except Exception as e:
            # Skip games with errors
            pass
    
    def _build_book_from_stats(self):
        """
        Convert position statistics into weighted opening book.
        """
        print("Converting statistics to opening book...")
        
        for fen, moves in self.position_stats.items():
            # Filter moves by minimum threshold
            valid_moves = [(move, count) for move, count in moves.items() 
                          if count >= self.min_games_threshold]
            
            if not valid_moves:
                continue
            
            # Sort by popularity
            valid_moves.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 8 moves to avoid huge book
            top_moves = valid_moves[:8]
            
            # Convert to weights (use square root to reduce dominance of very popular moves)
            import math
            weighted_moves = []
            for move, count in top_moves:
                # Weight formula: base weight + bonus for popularity
                weight = max(1, int(math.sqrt(count) * 2))
                weighted_moves.append((move, weight))
            
            self.book[fen] = weighted_moves
    
    def get_book_move(self, fen: str, move_number: int) -> Optional[str]:
        """
        Get a move from the opening book for the given position.
        
        Args:
            fen: The current position in FEN notation
            move_number: The current move number (1-based)
            
        Returns:
            A chess move in algebraic notation, or None if not in book
        """
        if move_number > self.max_opening_moves:
            return None
        
        normalized_fen = self._normalize_fen(fen)
        
        if normalized_fen not in self.book:
            return None
        
        moves_with_weights = self.book[normalized_fen]
        return self._select_weighted_move(moves_with_weights)
    
    def _normalize_fen(self, fen: str) -> str:
        """Normalize FEN for book storage."""
        parts = fen.split()
        if len(parts) >= 4:
            return " ".join(parts[:4])
        return fen
    
    def _select_weighted_move(self, moves_with_weights: List[Tuple[str, int]]) -> str:
        """Select a move based on weights using weighted random selection."""
        if not moves_with_weights:
            return None
        
        moves, weights = zip(*moves_with_weights)
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(moves)
        
        rand_num = random.randint(1, total_weight)
        cumulative = 0
        
        for move, weight in moves_with_weights:
            cumulative += weight
            if rand_num <= cumulative:
                return move
        
        return moves[0]
    
    def get_position_info(self, fen: str) -> Dict:
        """Get detailed information about a position."""
        normalized_fen = self._normalize_fen(fen)
        
        if normalized_fen in self.book:
            moves = self.book[normalized_fen]
            total_games = sum(self.position_stats[normalized_fen].values()) if normalized_fen in self.position_stats else 0
            
            return {
                'in_book': True,
                'available_moves': len(moves),
                'moves': moves,
                'total_games': total_games
            }
        else:
            return {'in_book': False}
    
    def save_book(self, filename: str):
        """Save the opening book to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.book, f, indent=2)
        print(f"Opening book saved to: {filename}")
    
    def load_book(self, filename: str):
        """Load the opening book from JSON."""
        try:
            with open(filename, 'r') as f:
                self.book = json.load(f)
            print(f"Opening book loaded from: {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error loading book: {e}")
    
    def get_book_stats(self) -> Dict:
        """Get statistics about the opening book."""
        if not self.book:
            return {"positions": 0, "total_moves": 0}
        
        total_moves = sum(len(moves) for moves in self.book.values())
        
        # Get depth distribution
        depth_counts = defaultdict(int)
        for fen in self.book.keys():
            # Estimate depth from position (crude but works)
            parts = fen.split()
            if len(parts) >= 6:
                try:
                    move_num = int(parts[5])
                    depth_counts[move_num] += 1
                except:
                    pass
        
        return {
            "positions": len(self.book),
            "total_moves": total_moves,
            "avg_moves_per_position": round(total_moves / len(self.book), 2),
            "depth_distribution": dict(depth_counts)
        }

# Example usage
"""
def main():
    # Create opening book builder
    book_builder = OpeningBook()
    
    #Example 1: Build from single Lichess database file
    book_builder.build_from_lichess_pgn(
        "Games\\lichess_db_standard_rated_2016-07.pgn.zst",
        max_games=3000000,  # Process first 100k games
        min_rating=1200,   # Only games with avg rating >= 1800
        time_controls=['blitz', 'rapid', 'classical']
    )
    
    #Example 2: Build from multiple files (recommended)
    # pgn_files = [
    #     "Games\\lichess_db_standard_rated_2016-07.pgn.zst",
    #     "Games\\lichess_db_standard_rated_2016-08.pgn.zst"
    # ]
    # book_builder.build_from_multiple_files(
    #     pgn_files,
    #     min_rating=1300,
    #     time_controls=['rapid', 'classical']
    # )
    
    #Save the book
    book_builder.save_book("chess_opening_book.json")
    
    #Print statistics
    stats = book_builder.get_book_stats()
    print(f"Book statistics: {stats}")
    
    #Test usage
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    move = book_builder.get_book_move(starting_fen, 1)
    print(f"Opening move: {move}")
    
    #Show position info
    info = book_builder.get_position_info(starting_fen)
    print(f"Position info: {info}")
    
    print("Opening book builder ready!")

main()
"""