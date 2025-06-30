import random
import math
from NodeEdge import *
from utils import *

class MCTS:
    """
    Monte Carlo Tree Search implementation for chess using a trained neural network.
    Designed to work with a model that outputs position evaluation and move probabilities.
    """
    
    def __init__(self, model, c_puct: float = 1.0, stochastic_selection: bool = False):
        """
        Initialize MCTS with a trained neural network.
        
        Args:
            model: Trained neural network that takes FEN and returns (value, move_probs)
                  - value: position evaluation from current player's perspective [-1, 1]
                  - move_probs: dict mapping chess.Move to probabilities
            c_puct (float): Exploration constant for UCB calculation
            stochastic_selection (bool): If True, use stochastic selection based on UCB scores
                                       If False, use deterministic selection (argmax)
        """
        self.model = model
        self.c_puct = c_puct
        self.stochastic_selection = stochastic_selection
        self.root = None
    
    def search(self, root_node: Node, num_simulations: int) -> Edge:
        """
        Run MCTS search for the specified number of simulations.
        
        Args:
            root_node (Node): Root position to search from
            num_simulations (int): Number of MCTS simulations to run
            
        Returns:
            Edge: Best move found by MCTS
        """
        self.root = root_node
        
        for _ in range(num_simulations):
            # 1. Selection: traverse tree to leaf node
            leaf_node, path = self._select(root_node)
            
            # 2. Expansion & Evaluation: expand leaf if not terminal and evaluate
            value = 0.0
            if self._is_terminal(leaf_node):
                # Terminal position - get exact value
                value = self._get_terminal_value(leaf_node)
            else:
                # 2. Expansion: expand the leaf node
                self._expand(leaf_node)
                
                # 3. Evaluation: evaluate the expanded position
                value = self._evaluate(leaf_node)
            
            # 4. Backpropagation: update values up the tree
            self._backpropagate(path, value)
        
        # Return the best move (most visited edge from root)
        return self._get_best_move(root_node)
    
    def _make_move_and_create_child(self, edge: Edge) -> Node:
        """
        Create a child node by making the move represented by the edge.
        
        Args:
            edge (Edge): Edge representing the move to make
            
        Returns:
            Node: Child node after making the move
        """
        board = edge.input_node.get_board()
        board.push(edge.move)
        
        child_node = Node(board.fen(), board.turn)
        edge.set_output_node(child_node)
    
        return child_node
    
    def _select(self, node: Node) -> tuple[Node, List[Edge]]:
        """
        Selection phase: traverse from root to leaf using UCB.
        Can use either deterministic (argmax) or stochastic selection.
        
        Returns:
            tuple: (leaf_node, path_of_edges)
        """
        path = []
        current_node = node
        
        while current_node.is_expanded() and not self._is_terminal(current_node):
            if self.stochastic_selection:
                # Stochastic selection based on UCB scores
                best_edge = self._select_edge_stochastic(current_node)
            else:
                # Deterministic selection (argmax)
                best_edge = self._select_edge_deterministic(current_node)
            
            path.append(best_edge)
            
            # Move to child node, create if doesn't exist
            if best_edge.output_node is None:
                best_edge.output_node = self._make_move_and_create_child(best_edge)
            
            current_node = best_edge.output_node
        
        return current_node, path
    
    def _select_edge_deterministic(self, node: Node) -> Edge:
        """
        Select edge with highest UCB score (deterministic).
        
        Args:
            node: Node to select edge from
            
        Returns:
            Edge: Edge with highest UCB score
        """
        return max(node.edges, 
                  key=lambda e: e.get_ucb_score(node.visit_count, self.c_puct))
    
    def _select_edge_stochastic(self, node: Node) -> Edge:
        """
        Select edge stochastically based on UCB scores.
        Uses softmax to convert UCB scores to probabilities.
        
        Args:
            node: Node to select edge from
            
        Returns:
            Edge: Stochastically selected edge
        """
        
        if not node.edges:
            raise ValueError("No edges to select from")
        
        # Calculate UCB scores
        ucb_scores = [edge.get_ucb_score(node.visit_count, self.c_puct) 
                     for edge in node.edges]
        
        # Handle infinite UCB scores (unvisited nodes)
        if any(score == float('inf') for score in ucb_scores):
            # If any edge has infinite UCB (unvisited), select randomly among them
            unvisited_edges = [edge for edge, score in zip(node.edges, ucb_scores) 
                              if score == float('inf')]
            return random.choice(unvisited_edges)
        
        # Apply softmax to convert UCB scores to probabilities
        # Use temperature to control randomness
        temperature = 1.0  # You can make this configurable if needed
        
        # Subtract max for numerical stability
        max_score = max(ucb_scores)
        exp_scores = [math.exp((score - max_score) / temperature) for score in ucb_scores]
        
        total = sum(exp_scores)
        probabilities = [exp_score / total for exp_score in exp_scores]
        
        # Sample based on probabilities
        r = random.random()
        cumulative = 0.0
        for edge, prob in zip(node.edges, probabilities):
            cumulative += prob
            if r <= cumulative:
                return edge
        
        # Fallback (should rarely happen due to floating point precision)
        return node.edges[-1]
    
    def _expand(self, node: Node) -> None:
        """
        Expansion phase: add all legal moves as edges using neural network priors.
        """
        if node.is_expanded() or self._is_terminal(node):
            return
        
        board = node.get_board()
        
        # Get move probabilities from the neural network
        _, move_probs = self._get_network_output(board)
        
        # Add edges for all legal moves with their prior probabilities
        for move in board.legal_moves:
            prior = move_probs.get(move, 1e-8)  # Small epsilon for unseen moves
            edge = Edge(node, None, move, prior)
            node.add_edge(edge)
    
    def _evaluate(self, node: Node) -> float:
        """
        Evaluation phase: get position value from neural network.
        
        Returns:
            float: Value from current player's perspective [-1, 1]
        """
        board = node.get_board()
        
        # Check for terminal positions first
        if self._is_terminal(node):
            return self._get_terminal_value(node)
        
        # Get position evaluation from neural network
        value, _ = self._get_network_output(board)
        return value
    
    def _get_network_output(self, board: chess.Board) -> tuple[float, dict]:
        """
        Get neural network output for a chess position.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            tuple: (position_value, move_probabilities_dict)
        """
        # Convert board to FEN for model input
        fen = board.fen()
        
        # Get model prediction
        # Assuming your model has a method like predict() that takes FEN
        # and returns (value, move_probs) where move_probs is dict mapping moves to probs
        policy_output, eval_output = self.model.predict(fen)

        # Remove batch dimension if present
        if policy_output.dim() == 2:
            policy_values = policy_output.squeeze(0)
        else:
            policy_values = policy_output
        
        move_dict = {}
        
        for encoded_move in range(4672):
            try:
                # Decode the move using your decode function
                chess_move = decode_move(encoded_move)
                # Get the corresponding value from the policy output
                move_value = policy_values[encoded_move].item()
                move_dict[chess_move] = move_value
            except ValueError:
                # Skip invalid moves that can't be decoded
                continue
        
        return eval_output, move_dict
    
    def _get_terminal_value(self, node: Node) -> float:
        """
        Get exact value for terminal positions.
        
        Returns:
            float: Exact game result from current player's perspective
        """
        board = node.get_board()
        
        if board.is_checkmate():
            # Current player is checkmated - they lose
            return -1.0
        elif board.is_stalemate() or board.is_insufficient_material() or board.is_seventyfive_moves() or board.is_fivefold_repetition():
            # Draw
            return 0.0
        else:
            # Should not reach here if _is_terminal() is correct
            raise ValueError("Position is not terminal")
    
    def _backpropagate(self, path: List[Edge], value: float) -> None:
        """
        Backpropagation phase: update visit counts and values up the tree.
        
        Args:
            path: List of edges from root to leaf
            value: Value to backpropagate (from leaf node's perspective)
        """
        # The value needs to be flipped for each level (alternating players)
        current_value = value
        
        # Update edges in reverse order (from leaf to root)
        for edge in reversed(path):
            edge.visit_count += 1
            edge.value += current_value
            
            # Update input node
            edge.input_node.visit_count += 1
            edge.input_node.value += current_value
            
            # Flip value for opponent
            current_value = -current_value
    
    def _is_terminal(self, node: Node) -> bool:
        """Check if the node represents a terminal game state."""
        board = node.get_board()
        return board.is_game_over()
    
    def _get_best_move(self, node: Node) -> Edge:
        """
        Get the best move from a node (most visited edge).
        
        Returns:
            Edge: Most visited edge from the node
        """
        if not node.edges:
            raise ValueError("No moves available from this node")
        
        return max(node.edges, key=lambda e: e.visit_count)
    
    def get_move_probabilities(self, node: Node, temperature: float = 1.0) -> dict:
        """
        Get move probabilities based on visit counts (for training data generation).
        
        Args:
            node: Node to get probabilities for
            temperature: Temperature for softmax (lower = more deterministic)
            
        Returns:
            dict: Mapping from chess.Move to probability
        """
        if not node.edges:
            return {}
        
        if temperature == 0:
            # Deterministic: choose most visited move
            best_edge = max(node.edges, key=lambda e: e.visit_count)
            return {edge.move: 1.0 if edge == best_edge else 0.0 for edge in node.edges}
        
        # Apply temperature to visit counts
        import math
        visit_counts = [edge.visit_count for edge in node.edges]
        
        if all(count == 0 for count in visit_counts):
            # Uniform if no visits
            prob = 1.0 / len(node.edges)
            return {edge.move: prob for edge in node.edges}
        
        # Apply temperature scaling
        scaled_counts = [count ** (1.0 / temperature) if count > 0 else 1e-10 
                        for count in visit_counts]
        
        total = sum(scaled_counts)
        probabilities = [count / total for count in scaled_counts]
        
        return {edge.move: prob for edge, prob in zip(node.edges, probabilities)}
    
    def get_search_statistics(self, node: Node) -> dict:
        """
        Get detailed statistics about the search from a node.
        Useful for analysis and debugging.
        
        Returns:
            dict: Search statistics
        """
        if not node.edges:
            return {}
        
        stats = {
            'total_visits': node.visit_count,
            'total_edges': len(node.edges),
            'moves': []
        }
        
        for edge in sorted(node.edges, key=lambda e: e.visit_count, reverse=True):
            move_stats = {
                'move': str(edge.move),
                'visits': edge.visit_count,
                'prior': edge.prior_probability,
                'q_value': edge.get_q_value(),
                'ucb_score': edge.get_ucb_score(node.visit_count, self.c_puct),
                'visit_percentage': edge.visit_count / max(node.visit_count, 1) * 100
            }
            stats['moves'].append(move_stats)
        
        return stats