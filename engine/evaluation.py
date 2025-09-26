"""
Chess position evaluation - imports from your existing files
"""
import chess
import logging

logger = logging.getLogger(__name__)

# Import your existing evaluators
try:
    # Try to import from different possible locations
    try:
        from .eval_ml import MLEvaluator
    except ImportError:
        try:
            from eval_ml import MLEvaluator
        except ImportError:
            from ..eval_ml import MLEvaluator
except ImportError as e:
    logger.warning(f"Could not import MLEvaluator: {e}")
    MLEvaluator = None

def get_piece_value(piece_type):
    """Get the value of a piece"""
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0
    }
    return values.get(piece_type, 0.0)

def is_development_square(square, piece_type, color):
    """Check if piece is on a development square"""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    if piece_type in [chess.KNIGHT, chess.BISHOP]:
        if color == chess.WHITE:
            return rank >= 2  # Developed from back rank
        else:
            return rank <= 5  # Developed from back rank
    
    return False

class BaselineEvaluator:
    """Simple heuristic evaluator for chess positions - your Phase 1 version"""
    
    def __init__(self):
        self.piece_square_tables = self._init_piece_square_tables()
    
    def evaluate(self, board: chess.Board) -> float:
        """Evaluate position from the perspective of the side to move"""
        if board.is_checkmate():
            return -1000.0
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0
        
        # Calculate score for both sides
        white_score = self._evaluate_side(board, chess.WHITE)
        black_score = self._evaluate_side(board, chess.BLACK)
        
        # Return from perspective of side to move
        if board.turn == chess.WHITE:
            return white_score - black_score
        else:
            return black_score - white_score
    
    def _evaluate_side(self, board: chess.Board, color: bool) -> float:
        """Evaluate one side's position"""
        score = 0.0
        
        # Material count
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.color == color:
                score += get_piece_value(piece.piece_type)
        
        # Simple mobility
        current_turn = board.turn
        if current_turn == color:
            score += len(list(board.legal_moves)) * 0.05
        
        return score
    
    def _init_piece_square_tables(self):
        """Simple piece square tables"""
        return {
            chess.PAWN: [0] * 64,
            chess.KNIGHT: [0] * 64,
            chess.BISHOP: [0] * 64,
            chess.ROOK: [0] * 64,
            chess.QUEEN: [0] * 64,
            chess.KING: [0] * 64,
        }

# Use your ML evaluator if available, otherwise baseline
if MLEvaluator:
    DefaultEvaluator = MLEvaluator
else:
    DefaultEvaluator = BaselineEvaluator
    print("Warning: Using baseline evaluator - ML evaluator not available")
