"""
Feature extraction for chess positions.
Converts a chess board position into a feature vector for ML models.
"""
import chess
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

# Piece-Square Tables (simplified, from White's perspective)
# Values are in centipawns, will be normalized for ML
PST_PAWN = [
    0,  0,  0,  0,  0,  0,  0,  0,
   50, 50, 50, 50, 50, 50, 50, 50,
   10, 10, 20, 30, 30, 20, 10, 10,
    5,  5, 10, 25, 25, 10,  5,  5,
    0,  0,  0, 20, 20,  0,  0,  0,
    5, -5,-10,  0,  0,-10, -5,  5,
    5, 10, 10,-20,-20, 10, 10,  5,
    0,  0,  0,  0,  0,  0,  0,  0
]

PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
]

PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
]

PST_ROOK = [
    0,  0,  0,  0,  0,  0,  0,  0,
    5, 10, 10, 10, 10, 10, 10,  5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
   -5,  0,  0,  0,  0,  0,  0, -5,
    0,  0,  0,  5,  5,  0,  0,  0
]

PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
]

PST_KING_MID = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
]

PST_KING_END = [
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50
]

PST_TABLES = {
    chess.PAWN: PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK: PST_ROOK,
    chess.QUEEN: PST_QUEEN,
    chess.KING: PST_KING_MID  # We'll handle endgame separately
}

class ChessFeatureExtractor:
    """Extracts features from chess positions for ML models."""
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Return list of all feature names."""
        names = []
        
        # Material features
        for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen']:
            names.extend([f'my_{piece}_count', f'opp_{piece}_count'])
        
        names.extend(['material_diff', 'total_material'])
        
        # Mobility features
        names.extend(['my_mobility', 'opp_mobility', 'mobility_diff'])
        
        # King safety features
        names.extend(['my_king_safety', 'opp_king_safety', 'my_castled', 'opp_castled'])
        
        # Pawn structure features
        names.extend(['my_doubled_pawns', 'opp_doubled_pawns',
                     'my_isolated_pawns', 'opp_isolated_pawns',
                     'my_passed_pawns', 'opp_passed_pawns'])
        
        # Piece-square table features
        for piece in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
            names.extend([f'my_{piece}_pst', f'opp_{piece}_pst'])
        
        # Positional features
        names.extend(['center_control', 'development', 'is_endgame'])
        
        # Game phase
        names.append('ply_number')
        
        return names
    
    def extract_features(self, board: chess.Board, ply: int = 0) -> Dict[str, float]:
        """Extract all features from a chess position."""
        features = {}
        
        # Determine perspective (always from side-to-move viewpoint)
        my_color = board.turn
        opp_color = not my_color
        
        # Material features
        material_features = self._extract_material_features(board, my_color, opp_color)
        features.update(material_features)
        
        # Mobility features
        mobility_features = self._extract_mobility_features(board, my_color, opp_color)
        features.update(mobility_features)
        
        # King safety features
        king_safety_features = self._extract_king_safety_features(board, my_color, opp_color)
        features.update(king_safety_features)
        
        # Pawn structure features
        pawn_features = self._extract_pawn_structure_features(board, my_color, opp_color)
        features.update(pawn_features)
        
        # Piece-square table features
        pst_features = self._extract_pst_features(board, my_color, opp_color)
        features.update(pst_features)
        
        # Positional features
        positional_features = self._extract_positional_features(board, my_color, opp_color)
        features.update(positional_features)
        
        # Game phase
        features['ply_number'] = ply
        
        return features
    
    def _extract_material_features(self, board: chess.Board, my_color: bool, opp_color: bool) -> Dict[str, float]:
        """Extract material-related features."""
        features = {}
        material_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3.2, 
                          chess.ROOK: 5, chess.QUEEN: 9}
        
        my_material = 0
        opp_material = 0
        
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            my_count = len(board.pieces(piece_type, my_color))
            opp_count = len(board.pieces(piece_type, opp_color))
            
            piece_name = chess.piece_name(piece_type)
            features[f'my_{piece_name}_count'] = my_count
            features[f'opp_{piece_name}_count'] = opp_count
            
            my_material += my_count * material_values[piece_type]
            opp_material += opp_count * material_values[piece_type]
        
        features['material_diff'] = my_material - opp_material
        features['total_material'] = my_material + opp_material
        
        return features
    
    def _extract_mobility_features(self, board: chess.Board, my_color: bool, opp_color: bool) -> Dict[str, float]:
        """Extract mobility-related features."""
        # Current turn is my turn
        my_mobility = len(list(board.legal_moves))
        
        # Switch turn to calculate opponent mobility
        board.push(chess.Move.null())  # Null move
        try:
            opp_mobility = len(list(board.legal_moves))
        except:
            opp_mobility = 0
        board.pop()  # Restore original position
        
        return {
            'my_mobility': my_mobility,
            'opp_mobility': opp_mobility,
            'mobility_diff': my_mobility - opp_mobility
        }
    
    def _extract_king_safety_features(self, board: chess.Board, my_color: bool, opp_color: bool) -> Dict[str, float]:
        """Extract king safety features."""
        features = {}
        
        my_king_square = board.king(my_color)
        opp_king_square = board.king(opp_color)
        
        # Castling rights and castled status
        my_castled = self._has_castled(board, my_color)
        opp_castled = self._has_castled(board, opp_color)
        
        features['my_castled'] = 1.0 if my_castled else 0.0
        features['opp_castled'] = 1.0 if opp_castled else 0.0
        
        # King safety score (based on pawn shield, open files, etc.)
        my_king_safety = self._calculate_king_safety(board, my_king_square if my_king_square is not None else -1, my_color)
        opp_king_safety = self._calculate_king_safety(board, opp_king_square if opp_king_square is not None else -1, opp_color)
        
        features['my_king_safety'] = my_king_safety
        features['opp_king_safety'] = opp_king_safety
        
        return features
    
    def _has_castled(self, board: chess.Board, color: bool) -> bool:
        """Check if a side has castled (heuristic based on king position)."""
        king_square = board.king(color)
        if king_square is None:
            return False
        
        # Check if king is on castled squares
        if color == chess.WHITE:
            return chess.square_file(king_square) in [2, 6]  # 2 = c-file, 6 = g-file
        else:
            return chess.square_file(king_square) in [2, 6]  # 2 = c-file, 6 = g-file
    
    def _calculate_king_safety(self, board: chess.Board, king_square: int, color: bool) -> float:
        """Calculate a king safety score."""
        if king_square is None:
            return -10.0  # Very unsafe if no king
        
        safety_score = 0.0
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        # Pawn shield bonus
        shield_files = [max(0, king_file-1), king_file, min(7, king_file+1)]
        for file_idx in shield_files:
            # Check for pawns in front of king
            if color == chess.WHITE:
                shield_squares = [chess.square(file_idx, king_rank + r) 
                                for r in [1, 2] if king_rank + r < 8]
            else:
                shield_squares = [chess.square(file_idx, king_rank - r) 
                                for r in [1, 2] if king_rank - r >= 0]
            
            for sq in shield_squares:
                if board.piece_at(sq) == chess.Piece(chess.PAWN, color):
                    safety_score += 0.5
        
        # Penalty for open files near king
        for file_idx in shield_files:
            file_pawns = len([sq for sq in chess.SquareSet(chess.BB_FILES[file_idx]) 
                            if (piece := board.piece_at(sq)) is not None and piece.piece_type == chess.PAWN])
            if file_pawns == 0:  # Open file
                safety_score -= 1.0
        
        return safety_score
    
    def _extract_pawn_structure_features(self, board: chess.Board, my_color: bool, opp_color: bool) -> Dict[str, float]:
        """Extract pawn structure features."""
        features = {}
        
        my_pawns = board.pieces(chess.PAWN, my_color)
        opp_pawns = board.pieces(chess.PAWN, opp_color)
        
        # Count doubled pawns
        features['my_doubled_pawns'] = self._count_doubled_pawns(my_pawns)
        features['opp_doubled_pawns'] = self._count_doubled_pawns(opp_pawns)
        
        # Count isolated pawns
        features['my_isolated_pawns'] = self._count_isolated_pawns(my_pawns)
        features['opp_isolated_pawns'] = self._count_isolated_pawns(opp_pawns)
        
        # Count passed pawns
        features['my_passed_pawns'] = self._count_passed_pawns(board, my_pawns, my_color, opp_pawns)
        features['opp_passed_pawns'] = self._count_passed_pawns(board, opp_pawns, opp_color, my_pawns)
        
        return features
    
    def _count_doubled_pawns(self, pawns: chess.SquareSet) -> float:
        """Count doubled pawns."""
        file_counts = [0] * 8
        for square in pawns:
            file_counts[chess.square_file(square)] += 1
        return sum(max(0, count - 1) for count in file_counts)
    
    def _count_isolated_pawns(self, pawns: chess.SquareSet) -> float:
        """Count isolated pawns."""
        files_with_pawns = set(chess.square_file(sq) for sq in pawns)
        isolated = 0
        
        for file_idx in files_with_pawns:
            has_neighbor = False
            for adj_file in [file_idx - 1, file_idx + 1]:
                if 0 <= adj_file <= 7 and adj_file in files_with_pawns:
                    has_neighbor = True
                    break
            if not has_neighbor:
                isolated += 1
        
        return isolated
    
    def _count_passed_pawns(self, board: chess.Board, my_pawns: chess.SquareSet, 
                           my_color: bool, opp_pawns: chess.SquareSet) -> float:
        """Count passed pawns (simplified check)."""
        passed = 0
        
        for pawn_sq in my_pawns:
            file_idx = chess.square_file(pawn_sq)
            rank = chess.square_rank(pawn_sq)
            
            # Check if any opponent pawn can stop this pawn
            is_passed = True
            check_files = [file_idx - 1, file_idx, file_idx + 1]
            
            for check_file in check_files:
                if not (0 <= check_file <= 7):
                    continue
                
                for opp_sq in opp_pawns:
                    if chess.square_file(opp_sq) == check_file:
                        opp_rank = chess.square_rank(opp_sq)
                        # Check if opponent pawn is ahead of our pawn
                        if my_color == chess.WHITE and opp_rank > rank:
                            is_passed = False
                            break
                        elif my_color == chess.BLACK and opp_rank < rank:
                            is_passed = False
                            break
                
                if not is_passed:
                    break
            
            if is_passed:
                passed += 1
        
        return passed
    
    def _extract_pst_features(self, board: chess.Board, my_color: bool, opp_color: bool) -> Dict[str, float]:
        """Extract piece-square table features."""
        features = {}
        is_endgame = self._is_endgame(board)
        
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            piece_name = chess.piece_name(piece_type)
            
            # Get appropriate PST
            if piece_type == chess.KING:
                pst = PST_KING_END if is_endgame else PST_KING_MID
            else:
                pst = PST_TABLES.get(piece_type, [0] * 64)
            
            my_pst_score = 0.0
            opp_pst_score = 0.0
            
            # Calculate PST scores
            my_pieces = board.pieces(piece_type, my_color)
            opp_pieces = board.pieces(piece_type, opp_color)
            
            for square in my_pieces:
                # Flip square for black pieces
                pst_idx = square if my_color == chess.WHITE else chess.square_mirror(square)
                my_pst_score += pst[pst_idx]
            
            for square in opp_pieces:
                # Flip square for black pieces
                pst_idx = square if opp_color == chess.WHITE else chess.square_mirror(square)
                opp_pst_score += pst[pst_idx]
            
            # Normalize to reasonable range
            features[f'my_{piece_name}_pst'] = my_pst_score / 100.0
            features[f'opp_{piece_name}_pst'] = opp_pst_score / 100.0
        
        return features
    
    def _extract_positional_features(self, board: chess.Board, my_color: bool, opp_color: bool) -> Dict[str, float]:
        """Extract positional features."""
        features = {}
        
        # Center control (e4, e5, d4, d5)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        center_control = 0.0
        
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                if piece.color == my_color:
                    center_control += 1.0
                else:
                    center_control -= 1.0
            
            # Also check attacks on center squares
            my_attacks = len(board.attackers(my_color, square))
            opp_attacks = len(board.attackers(opp_color, square))
            center_control += 0.1 * (my_attacks - opp_attacks)
        
        features['center_control'] = center_control
        
        # Development score (pieces off back rank)
        development = 0.0
        back_rank = 0 if my_color == chess.WHITE else 7  # 0=rank1, 7=rank8
        
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            my_pieces = board.pieces(piece_type, my_color)
            for square in my_pieces:
                if chess.square_rank(square) != back_rank:
                    development += 1.0
        
        features['development'] = development
        
        # Endgame indicator
        features['is_endgame'] = 1.0 if self._is_endgame(board) else 0.0
        
        return features
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Determine if position is in endgame."""
        # Simple heuristic: endgame if queens are off or low material
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        
        if queens == 0:
            return True
        
        # Count total minor and major pieces
        minors_majors = 0
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            minors_majors += len(board.pieces(piece_type, chess.WHITE))
            minors_majors += len(board.pieces(piece_type, chess.BLACK))
        
        return minors_majors <= 6  # Arbitrary threshold
    
    def extract_features_from_fen(self, fen: str, ply: int = 0) -> Dict[str, float]:
        """Extract features from a FEN string."""
        try:
            board = chess.Board(fen)
            return self.extract_features(board, ply)
        except ValueError as e:
            logger.error(f"Invalid FEN: {fen}, error: {e}")
            return {name: 0.0 for name in self.feature_names}
    
    def feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order."""
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.feature_names)

def test_feature_extraction():
    """Test feature extraction on standard positions."""
    extractor = ChessFeatureExtractor()
    
    # Test starting position
    start_board = chess.Board()
    features = extractor.extract_features(start_board)
    
    print(f"Feature count: {len(features)}")
    print(f"Expected feature count: {extractor.get_feature_count()}")
    print("\nStarting position features:")
    
    for name, value in sorted(features.items()):
        if abs(value) > 0.001:  # Only show non-zero features
            print(f"  {name}: {value:.3f}")
    
    # Test feature vector conversion
    feature_vec = extractor.feature_vector(features)
    print(f"\nFeature vector shape: {feature_vec.shape}")
    print(f"Non-zero elements: {np.count_nonzero(feature_vec)}")
    
    # Test a more complex position
    complex_fen = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"
    complex_features = extractor.extract_features_from_fen(complex_fen, ply=8)
    
    print(f"\nComplex position (after 1.e4 e5 2.Nf3 Nf6) features:")
    for name, value in sorted(complex_features.items()):
        if abs(value) > 0.001:
            print(f"  {name}: {value:.3f}")

if __name__ == "__main__":
    test_feature_extraction()