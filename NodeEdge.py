import chess
from typing import List, Optional


class Node:
    """
    Represents a node in the MCTS tree for chess positions.
    """
    
    def __init__(self, fen: str, is_white_to_move: bool):
        """
        Initialize a new node.
        
        Args:
            fen (str): Chess position in FEN notation
            is_white_to_move (bool): True if white to move, False if black to move
        """
        self.fen = fen
        self.is_white_to_move = is_white_to_move
        self.edges: List['Edge'] = []
        self.visit_count = 0
        self.value = 0.0
    
    def add_edge(self, edge: 'Edge') -> None:
        """Add an edge to this node's list of connected edges."""
        self.edges.append(edge)
    
    def is_expanded(self) -> bool:
        """Check if this node has been expanded (has edges)."""
        return len(self.edges) > 0
    
    def get_board(self) -> chess.Board:
        """Get a chess.Board object from the FEN position."""
        return chess.Board(self.fen)
    
    def __str__(self) -> str:
        return f"Node(FEN: {self.fen}, White to move: {self.is_white_to_move}, Visits: {self.visit_count}, Value: {self.value:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()


class Edge:
    """
    Represents an edge (chess move) in the MCTS tree.
    """
    
    def __init__(self, input_node: Node, output_node: Optional[Node], move: chess.Move, prior_probability: float):
        """
        Initialize a new edge.
        
        Args:
            input_node (Node): The node from which this move is made
            output_node (Node, optional): The resulting node after the move (can be None initially)
            move (chess.Move): The chess move represented by this edge
            prior_probability (float): Prior probability for this move (from neural network)
        """
        self.input_node = input_node
        self.output_node = output_node
        self.move = move
        self.prior_probability = prior_probability
        self.visit_count = 0
        self.value = 0.0
    
    def set_output_node(self, node: Node) -> None:
        """Set the output node for this edge."""
        self.output_node = node
    
    def get_q_value(self) -> float:
        """
        Get the Q-value (average value) for this edge.
        Returns 0 if never visited.
        """
        if self.visit_count == 0:
            return 0.0
        return self.value / self.visit_count
    
    def get_ucb_score(self, parent_visit_count: int, c_puct: float = 1.0) -> float:
        """
        Calculate the UCB (Upper Confidence Bound) score for this edge.
        Used for selecting which move to explore next in MCTS.
        
        Args:
            parent_visit_count (int): Visit count of the parent node
            c_puct (float): Exploration constant
            
        Returns:
            float: UCB score for this edge
        """
        if parent_visit_count == 0:
            return float('inf')
        
        exploration_term = c_puct * self.prior_probability * (parent_visit_count ** 0.5) / (1 + self.visit_count)
        return self.get_q_value() + exploration_term
    
    def __str__(self) -> str:
        return f"Edge(Move: {self.move}, Prior: {self.prior_probability:.3f}, Visits: {self.visit_count}, Value: {self.value:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()


# Example usage and utility functions
def create_root_node(board: chess.Board) -> Node:
    """
    Create a root node from a chess board position.
    
    Args:
        board (chess.Board): Current chess position
        
    Returns:
        Node: Root node for MCTS
    """
    return Node(board.fen(), board.turn)
