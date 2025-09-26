"""
Utility functions for the chess engine
"""
import chess
import time
from typing import Optional

def fen_to_board(fen: str) -> chess.Board:
    """Convert FEN string to python-chess Board object"""
    try:
        return chess.Board(fen)
    except ValueError as e:
        raise ValueError(f"Invalid FEN: {fen}. Error: {e}")

def board_to_fen(board: chess.Board) -> str:
    """Convert Board object to FEN string"""
    return board.fen()

def uci_to_move(board: chess.Board, uci: str) -> Optional[chess.Move]:
    """Convert UCI string to Move object, return None if invalid"""
    try:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
        return None
    except ValueError:
        return None

def move_to_uci(move: chess.Move) -> str:
    """Convert Move object to UCI string"""
    return move.uci()

class Timer:
    """Simple timer for time control"""
    def __init__(self):
        self.start_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        return (time.time() - self.start_time) * 1000
    
    def is_time_up(self, limit_ms: float) -> bool:
        return self.elapsed_ms() >= limit_ms

def get_piece_value(piece_type: int) -> float:
    """Get material value for piece type"""
    values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.2,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
        chess.KING: 0.0  # King is invaluable
    }
    return values.get(piece_type, 0.0)

def is_development_square(square: int, piece_type: int, color: bool) -> bool:
    """Check if piece is on a development square"""
    if piece_type not in [chess.KNIGHT, chess.BISHOP]:
        return False
    
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    # Good development squares for minor pieces
    if color == chess.WHITE:
        return rank in [2, 3] and file in [2, 3, 4, 5]  # c3, d3, e3, f3, etc.
    else:
        return rank in [4, 5] and file in [2, 3, 4, 5]  # c6, d6, e6, f6, etc.