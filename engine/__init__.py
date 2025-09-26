"""
Chess Engine Package
"""
from .eval_ml import BaselineEvaluator
from .search import ChessSearch, SearchResult
from .utils import fen_to_board, move_to_uci, uci_to_move, Timer

# Main engine class
import chess
from .search import ChessSearch
from .eval_ml import BaselineEvaluator
from .utils import fen_to_board, move_to_uci

class ChessEngine:
    """Main chess engine class"""
    
    def __init__(self):
        self.evaluator = BaselineEvaluator()
        self.search_engine = ChessSearch(self.evaluator)
    
    def get_best_move(self, fen: str, depth: int = 2, 
                     time_limit_ms: float = 400.0) -> dict:
        """
        Get the best move for a given position
        
        Args:
            fen: FEN string of position
            depth: Search depth (2-3 recommended)
            time_limit_ms: Time limit in milliseconds
            
        Returns:
            Dictionary with best move and analysis
        """
        try:
            board = fen_to_board(fen)
            result = self.search_engine.search(board, depth, time_limit_ms)
            return result.to_dict()
        except Exception as e:
            return {
                'error': str(e),
                'bestMove': None,
                'score': 0.0,
                'nodes': 0,
                'pv': [],
                'duration_ms': 0.0
            }
    
    def evaluate_position(self, fen: str) -> dict:
        """
        Evaluate a position without searching
        
        Args:
            fen: FEN string of position
            
        Returns:
            Dictionary with evaluation score
        """
        try:
            board = fen_to_board(fen)
            score = self.evaluator.evaluate(board)
            return {
                'score': round(score, 3),
                'fen': fen,
                'turn': 'white' if board.turn else 'black'
            }
        except Exception as e:
            return {
                'error': str(e),
                'score': 0.0
            }
    
    def is_legal_move(self, fen: str, move_uci: str) -> bool:
        """Check if a move is legal in the given position"""
        try:
            board = fen_to_board(fen)
            move = chess.Move.from_uci(move_uci)
            return move in board.legal_moves
        except:
            return False
    
    def make_move(self, fen: str, move_uci: str) -> dict:
        """
        Make a move and return the new position
        
        Args:
            fen: Current position FEN
            move_uci: Move in UCI format (e.g., 'e2e4')
            
        Returns:
            Dictionary with new FEN and move info
        """
        try:
            board = fen_to_board(fen)
            move = chess.Move.from_uci(move_uci)
            
            if move not in board.legal_moves:
                return {
                    'error': f'Illegal move: {move_uci}',
                    'fen': fen
                }
            
            # Make the move
            board.push(move)
            
            return {
                'fen': board.fen(),
                'move': move_uci,
                'legal': True,
                'check': board.is_check(),
                'checkmate': board.is_checkmate(),
                'stalemate': board.is_stalemate(),
                'game_over': board.is_game_over()
            }
        except Exception as e:
            return {
                'error': str(e),
                'fen': fen
            }

# Create a default engine instance
default_engine = ChessEngine()

# Export main classes
__all__ = [
    'ChessEngine',
    'BaselineEvaluator', 
    'ChessSearch',
    'SearchResult',
    'default_engine'
]