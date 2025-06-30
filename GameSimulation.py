import json
import chess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import random
from typing import List, Dict, Any, Optional
import threading
import time

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Position Simulator")
        self.root.geometry("900x700")
        
        # Chess data
        self.positions = []
        self.current_position_index = 0
        self.current_board = None
        self.game_moves = []
        self.current_move_index = 0
        self.is_simulating = False
        
        # Colors
        self.light_square = "#F0D9B5"
        self.dark_square = "#B58863"
        self.highlight_color = "#FFFF00"
        
        # Unicode chess pieces
        self.piece_symbols = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }
        
        self.setup_gui()
    
    def setup_gui(self):
        """Set up the graphical user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File loading
        ttk.Button(control_frame, text="Load JSON File", 
                  command=self.load_json_file).grid(row=0, column=0, padx=(0, 5))
        
        self.file_label = ttk.Label(control_frame, text="No file loaded")
        self.file_label.grid(row=0, column=1, padx=(5, 0))
        
        # Position controls
        position_frame = ttk.Frame(control_frame)
        position_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(position_frame, text="Position:").grid(row=0, column=0)
        
        ttk.Button(position_frame, text="◀◀", 
                  command=self.first_position).grid(row=0, column=1, padx=2)
        ttk.Button(position_frame, text="◀", 
                  command=self.prev_position).grid(row=0, column=2, padx=2)
        
        self.position_label = ttk.Label(position_frame, text="0 / 0")
        self.position_label.grid(row=0, column=3, padx=10)
        
        ttk.Button(position_frame, text="▶", 
                  command=self.next_position).grid(row=0, column=4, padx=2)
        ttk.Button(position_frame, text="▶▶", 
                  command=self.last_position).grid(row=0, column=5, padx=2)
        
        # Simulation controls
        sim_frame = ttk.Frame(control_frame)
        sim_frame.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        ttk.Button(sim_frame, text="Simulate Game", 
                  command=self.simulate_current_position).grid(row=0, column=0, padx=(0, 5))
        ttk.Button(sim_frame, text="Auto Play All", 
                  command=self.auto_play_all).grid(row=0, column=1, padx=5)
        ttk.Button(sim_frame, text="Stop", 
                  command=self.stop_simulation).grid(row=0, column=2, padx=5)
        
        # Move controls
        move_frame = ttk.Frame(control_frame)
        move_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        ttk.Label(move_frame, text="Moves:").grid(row=0, column=0)
        
        ttk.Button(move_frame, text="◀◀", 
                  command=self.first_move).grid(row=0, column=1, padx=2)
        ttk.Button(move_frame, text="◀", 
                  command=self.prev_move).grid(row=0, column=2, padx=2)
        
        self.move_label = ttk.Label(move_frame, text="0 / 0")
        self.move_label.grid(row=0, column=3, padx=10)
        
        ttk.Button(move_frame, text="▶", 
                  command=self.next_move).grid(row=0, column=4, padx=2)
        ttk.Button(move_frame, text="▶▶", 
                  command=self.last_move).grid(row=0, column=5, padx=2)
        
        # Chess board frame
        board_frame = ttk.LabelFrame(main_frame, text="Chess Board", padding="5")
        board_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Create chess board
        self.board_canvas = tk.Canvas(board_frame, width=400, height=400, bg="white")
        self.board_canvas.grid(row=0, column=0)
        
        # Information panel
        info_frame = ttk.LabelFrame(main_frame, text="Information", padding="5")
        info_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # FEN display
        ttk.Label(info_frame, text="FEN:").grid(row=0, column=0, sticky=tk.W)
        self.fen_text = tk.Text(info_frame, height=3, width=40, wrap=tk.WORD)
        self.fen_text.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Game status
        ttk.Label(info_frame, text="Game Status:").grid(row=2, column=0, sticky=tk.W)
        self.status_label = ttk.Label(info_frame, text="No game loaded", foreground="blue")
        self.status_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 10))
        
        # Move history
        ttk.Label(info_frame, text="Move History:").grid(row=4, column=0, sticky=tk.W)
        
        # Scrollable move history
        history_frame = ttk.Frame(info_frame)
        history_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        info_frame.rowconfigure(5, weight=1)
        
        self.move_history = tk.Listbox(history_frame, height=15)
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.move_history.yview)
        self.move_history.configure(yscrollcommand=scrollbar.set)
        
        self.move_history.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        # Speed control
        speed_frame = ttk.Frame(info_frame)
        speed_frame.grid(row=6, column=0, sticky=(tk.W, tk.E))
        
        ttk.Label(speed_frame, text="Speed:").grid(row=0, column=0)
        self.speed_var = tk.DoubleVar(value=1.0)
        self.speed_scale = ttk.Scale(speed_frame, from_=0.1, to=3.0, 
                                   variable=self.speed_var, orient=tk.HORIZONTAL)
        self.speed_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(5, 0))
        speed_frame.columnconfigure(1, weight=1)
        
        self.draw_empty_board()
    
    def draw_empty_board(self):
        """Draw an empty chess board."""
        self.board_canvas.delete("all")
        square_size = 50
        
        for row in range(8):
            for col in range(8):
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                color = self.light_square if (row + col) % 2 == 0 else self.dark_square
                self.board_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                
                # Add coordinates
                if col == 0:  # Rank numbers
                    self.board_canvas.create_text(x1 + 5, y1 + 5, text=str(8-row), 
                                                fill="black", font=("Arial", 8))
                if row == 7:  # File letters
                    self.board_canvas.create_text(x2 - 5, y2 - 5, text=chr(ord('a') + col), 
                                                fill="black", font=("Arial", 8))
    
    def draw_board(self, board: chess.Board, last_move: Optional[chess.Move] = None):
        """Draw the chess board with pieces."""
        self.draw_empty_board()
        square_size = 50
        
        # Highlight last move
        if last_move:
            from_square = last_move.from_square
            to_square = last_move.to_square
            
            for square in [from_square, to_square]:
                row = 7 - (square // 8)
                col = square % 8
                x1 = col * square_size
                y1 = row * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                
                self.board_canvas.create_rectangle(x1, y1, x2, y2, 
                                                 fill=self.highlight_color, outline="", stipple="gray50")
        
        # Draw pieces
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row = 7 - (square // 8)
                col = square % 8
                x = col * square_size + square_size // 2
                y = row * square_size + square_size // 2
                
                symbol = self.piece_symbols.get(piece.symbol(), piece.symbol())
                self.board_canvas.create_text(x, y, text=symbol, font=("Arial", 24), fill="black")
    
    def load_json_file(self):
        """Load chess positions from JSON file."""
        file_path = filedialog.askopenfilename(
            title="Select JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self.positions = []
                for obj in data:
                    if 'fen' in obj:
                        self.positions.append(obj['fen'])
                
                self.current_position_index = 0
                self.game_moves = []
                self.current_move_index = 0
                
                self.file_label.config(text=f"Loaded {len(self.positions)} positions")
                self.update_position_display()
                
                if self.positions:
                    self.load_position(0)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")
    
    def load_position(self, index: int):
        """Load a specific position."""
        if 0 <= index < len(self.positions):
            self.current_position_index = index
            self.current_board = chess.Board(self.positions[index])
            self.game_moves = []
            self.current_move_index = 0
            
            self.update_display()
    
    def update_display(self):
        """Update all display elements."""
        if self.current_board:
            # Update board
            last_move = None
            if self.game_moves and self.current_move_index > 0:
                last_move = self.game_moves[self.current_move_index - 1]
            
            self.draw_board(self.current_board, last_move)
            
            # Update FEN
            self.fen_text.delete(1.0, tk.END)
            self.fen_text.insert(1.0, self.current_board.fen())
            
            # Update status
            status = self.get_game_status()
            self.status_label.config(text=status)
            
            # Update move history
            self.update_move_history()
        
        self.update_position_display()
        self.update_move_display()
    
    def update_position_display(self):
        """Update position counter display."""
        if self.positions:
            self.position_label.config(text=f"{self.current_position_index + 1} / {len(self.positions)}")
        else:
            self.position_label.config(text="0 / 0")
    
    def update_move_display(self):
        """Update move counter display."""
        if self.game_moves:
            self.move_label.config(text=f"{self.current_move_index} / {len(self.game_moves)}")
        else:
            self.move_label.config(text="0 / 0")
    
    def update_move_history(self):
        """Update the move history display."""
        self.move_history.delete(0, tk.END)
        
        if self.game_moves:
            for i, move in enumerate(self.game_moves[:self.current_move_index]):
                move_num = (i // 2) + 1
                if i % 2 == 0:
                    self.move_history.insert(tk.END, f"{move_num}. {move.uci()}")
                else:
                    # Update the last entry to include black's move
                    last_entry = self.move_history.get(tk.END)
                    self.move_history.delete(tk.END)
                    self.move_history.insert(tk.END, f"{last_entry} {move.uci()}")
    
    def get_game_status(self):
        """Get current game status."""
        if not self.current_board:
            return "No game loaded"
        
        if self.current_board.is_checkmate():
            winner = "Black" if self.current_board.turn else "White"
            return f"Checkmate - {winner} wins!"
        elif self.current_board.is_stalemate():
            return "Stalemate - Draw"
        elif self.current_board.is_insufficient_material():
            return "Draw - Insufficient material"
        elif self.current_board.is_check():
            turn = "White" if self.current_board.turn else "Black"
            return f"{turn} is in check"
        else:
            turn = "White" if self.current_board.turn else "Black"
            return f"{turn} to move"
    
    def simulate_current_position(self):
        """Simulate a game from the current position."""
        if not self.current_board:
            messagebox.showwarning("Warning", "No position loaded!")
            return
        
        # Reset to initial position
        self.current_board = chess.Board(self.positions[self.current_position_index])
        self.game_moves = []
        self.current_move_index = 0
        
        # Simulate game
        max_moves = 50
        move_count = 0
        
        while not self.current_board.is_game_over() and move_count < max_moves:
            legal_moves = list(self.current_board.legal_moves)
            if not legal_moves:
                break
            
            move = random.choice(legal_moves)
            self.game_moves.append(move)
            self.current_board.push(move)
            move_count += 1
        
        # Reset board to initial position for playback
        self.current_board = chess.Board(self.positions[self.current_position_index])
        self.current_move_index = 0
        self.update_display()
        
    
    def auto_play_all(self):
        """Auto-play through all positions."""
        if not self.positions:
            messagebox.showwarning("Warning", "No positions loaded!")
            return
        
        self.is_simulating = True
        threading.Thread(target=self._auto_play_thread, daemon=True).start()
    
    def _auto_play_thread(self):
        """Thread function for auto-playing all positions."""
        for i in range(len(self.positions)):
            if not self.is_simulating:
                break
            
            self.root.after(0, self.load_position, i)
            self.root.after(100, self.simulate_current_position)
            
            # Wait for simulation speed
            time.sleep(2.0 / self.speed_var.get())
        
        self.is_simulating = False
    
    def stop_simulation(self):
        """Stop the auto-play simulation."""
        self.is_simulating = False
    
    # Position navigation methods
    def first_position(self):
        if self.positions:
            self.load_position(0)
    
    def prev_position(self):
        if self.positions and self.current_position_index > 0:
            self.load_position(self.current_position_index - 1)
    
    def next_position(self):
        if self.positions and self.current_position_index < len(self.positions) - 1:
            self.load_position(self.current_position_index + 1)
    
    def last_position(self):
        if self.positions:
            self.load_position(len(self.positions) - 1)
    
    # Move navigation methods
    def first_move(self):
        if self.game_moves:
            self.current_board = chess.Board(self.positions[self.current_position_index])
            self.current_move_index = 0
            self.update_display()
    
    def prev_move(self):
        if self.game_moves and self.current_move_index > 0:
            self.current_board = chess.Board(self.positions[self.current_position_index])
            self.current_move_index -= 1
            
            # Replay moves up to current index
            for i in range(self.current_move_index):
                self.current_board.push(self.game_moves[i])
            
            self.update_display()
    
    def next_move(self):
        if self.game_moves and self.current_move_index < len(self.game_moves):
            if self.current_move_index == 0:
                self.current_board = chess.Board(self.positions[self.current_position_index])
            
            self.current_board.push(self.game_moves[self.current_move_index])
            self.current_move_index += 1
            self.update_display()
    
    def last_move(self):
        if self.game_moves:
            self.current_board = chess.Board(self.positions[self.current_position_index])
            
            for move in self.game_moves:
                self.current_board.push(move)
            
            self.current_move_index = len(self.game_moves)
            self.update_display()

def main():
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()

main()