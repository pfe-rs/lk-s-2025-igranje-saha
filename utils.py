import chess
import numpy as np

def encode_move(move):
    """
    Encode a chess.Move to an integer representation.
    
    Move encoding follows AlphaZero's approach:
    - 64 squares * 73 move types = 4672 total possible moves
    - Move types: 56 queen-like moves + 8 knight moves + 9 underpromotion moves
    
    Args:
        move (chess.Move): The move to encode
        
    Returns:
        int: Encoded move (0-4671)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # Calculate direction and distance
    from_rank, from_file = divmod(from_square, 8)
    to_rank, to_file = divmod(to_square, 8)
    
    rank_diff = to_rank - from_rank
    file_diff = to_file - from_file
    
    # Check if it's a knight move
    if abs(rank_diff) == 2 and abs(file_diff) == 1:
        # Knight moves (8 possibilities)
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        move_type = 56 + knight_moves.index((rank_diff, file_diff))
    elif abs(rank_diff) == 1 and abs(file_diff) == 2:
        # Knight moves (8 possibilities)
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        move_type = 56 + knight_moves.index((rank_diff, file_diff))
    else:
        # Queen-like moves (56 possibilities)
        # 7 directions * 8 distances each = 56 moves
        # Directions: N, NE, E, SE, S, SW, W, NW
        
        if rank_diff == 0 and file_diff > 0:  # East
            direction = 2
            distance = file_diff
        elif rank_diff == 0 and file_diff < 0:  # West
            direction = 6
            distance = -file_diff
        elif rank_diff > 0 and file_diff == 0:  # North
            direction = 0
            distance = rank_diff
        elif rank_diff < 0 and file_diff == 0:  # South
            direction = 4
            distance = -rank_diff
        elif rank_diff > 0 and file_diff > 0 and rank_diff == file_diff:  # Northeast
            direction = 1
            distance = rank_diff
        elif rank_diff < 0 and file_diff > 0 and -rank_diff == file_diff:  # Southeast
            direction = 3
            distance = file_diff
        elif rank_diff < 0 and file_diff < 0 and rank_diff == file_diff:  # Southwest
            direction = 5
            distance = -rank_diff
        elif rank_diff > 0 and file_diff < 0 and rank_diff == -file_diff:  # Northwest
            direction = 7
            distance = rank_diff
        else:
            raise ValueError(f"Invalid move: {move}")
        
        if distance < 1 or distance > 7:
            raise ValueError(f"Invalid distance: {distance}")
        
        move_type = direction * 7 + (distance - 1)
    
    # Handle underpromotions (queen promotions are handled as regular queen moves above)
    if move.promotion and move.promotion != chess.QUEEN:
        # Underpromotion moves (excluding queen promotion)
        # 3 directions (N, NE, NW) * 3 piece types (R, B, N) = 9 moves
        promotion_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
        
        if move.promotion not in promotion_pieces:
            raise ValueError(f"Invalid promotion piece: {move.promotion}")
        
        promotion_idx = promotion_pieces.index(move.promotion)
        
        # Determine promotion direction
        if file_diff == -1:  # Capture left (NW)
            direction_idx = 2
        elif file_diff == 0:   # Push forward (N)
            direction_idx = 0
        elif file_diff == 1:   # Capture right (NE)
            direction_idx = 1
        else:
            raise ValueError(f"Invalid promotion move: {move}")
        
        move_type = 64 + direction_idx * 3 + promotion_idx
    
    return from_square * 73 + move_type


def decode_move(encoded_move):
    """
    Decode an integer to a chess.Move object.
    
    Args:
        encoded_move (int): Encoded move (0-4671)
        
    Returns:
        chess.Move: The decoded move
    """
    from_square = encoded_move // 73
    move_type = encoded_move % 73
    
    from_rank, from_file = divmod(from_square, 8)
    
    if move_type < 56:
        # Queen-like moves
        direction = move_type // 7
        distance = (move_type % 7) + 1
        
        # Direction mappings: N, NE, E, SE, S, SW, W, NW
        direction_deltas = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        rank_delta, file_delta = direction_deltas[direction]
        
        to_rank = from_rank + rank_delta * distance
        to_file = from_file + file_delta * distance
        
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Invalid decoded move: from={from_square}, to_rank={to_rank}, to_file={to_file}")
        
        to_square = to_rank * 8 + to_file
        return chess.Move(from_square, to_square)
    
    elif move_type < 64:
        # Knight moves
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        rank_delta, file_delta = knight_moves[move_type - 56]
        
        to_rank = from_rank + rank_delta
        to_file = from_file + file_delta
        
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Invalid decoded knight move: from={from_square}, to_rank={to_rank}, to_file={to_file}")
        
        to_square = to_rank * 8 + to_file
        return chess.Move(from_square, to_square)
    
    else:
        # Underpromotion moves
        promotion_type = move_type - 64
        direction_idx = promotion_type // 3
        piece_idx = promotion_type % 3
        
        promotion_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]
        promotion_piece = promotion_pieces[piece_idx]
        
        # Direction mappings for promotions: N, NE, NW
        if direction_idx == 0:      # North (push)
            file_delta = 0
        elif direction_idx == 1:    # Northeast (capture right)
            file_delta = 1
        elif direction_idx == 2:    # Northwest (capture left)
            file_delta = -1
        else:
            raise ValueError(f"Invalid promotion direction: {direction_idx}")
        
        # Determine promotion rank direction based on starting rank
        # White promotes from rank 6 to 7, black promotes from rank 1 to 0
        if from_rank == 6:  # White promotion
            to_rank = from_rank + 1
        elif from_rank == 1:  # Black promotion  
            to_rank = from_rank - 1
        else:
            raise ValueError(f"Invalid promotion from rank: {from_rank}")
        to_file = from_file + file_delta
        
        if not (0 <= to_rank <= 7 and 0 <= to_file <= 7):
            raise ValueError(f"Invalid decoded promotion move: from={from_square}, to_rank={to_rank}, to_file={to_file}")
        
        to_square = to_rank * 8 + to_file
        return chess.Move(from_square, to_square, promotion=promotion_piece)

def moves_dict_to_array(move_probs: dict, legal_moves: set = None) -> np.ndarray:
    """
    Convert move probabilities dictionary to array of 4672 values.
    
    Args:
        move_probs: Dictionary mapping chess.Move to probabilities
        legal_moves: Set of legal moves (optional, for masking)
        
    Returns:
        np.ndarray: Array of 4672 probability values
    """
    prob_array = np.zeros(4672, dtype=np.float32)
    
    for move, prob in move_probs.items():
        prob_array[encode_move(move)] = prob
    
    # Optionally mask illegal moves
    """
    if legal_moves is not None:
        mask = np.zeros(4672, dtype=bool)
        for move in legal_moves:
            if move in MOVE_TO_INDEX:
                mask[MOVE_TO_INDEX[move]] = True
        prob_array = prob_array * mask
        
        # Renormalize if needed
        total = prob_array.sum()
        if total > 0:
            prob_array = prob_array / total
    """
    return prob_array