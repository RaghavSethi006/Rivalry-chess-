"""
Chess search engine using minimax with alpha-beta pruning - Phase 4 upgrade
Enhanced with ML evaluation and improved search features
"""
import chess
import time
import math
from typing import List, Tuple, Optional
from .utils import Timer, move_to_uci

class SearchResult:
    """Container for search results - enhanced for Phase 4"""
    def __init__(self, best_move: Optional[chess.Move], score: float, 
                 nodes: int, pv: List[chess.Move], duration_ms: float,
                 depth_reached: int = 0):
        self.best_move = best_move
        self.score = score
        self.nodes = nodes
        self.pv = pv  # Principal variation
        self.duration_ms = duration_ms
        self.depth_reached = depth_reached
        
    def to_dict(self):
        return {
            'bestMove': move_to_uci(self.best_move) if self.best_move else None,
            'score': round(self.score, 3),
            'nodes': self.nodes,
            'pv': [move_to_uci(move) for move in self.pv],
            'duration_ms': round(self.duration_ms, 1),
            'depth': self.depth_reached
        }

class ChessSearch:
    """Enhanced chess search engine with ML evaluation and improved features"""
    
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.nodes_searched = 0
        self.timer = Timer()
        
        # Transposition table (simple dict cache)
        self.tt_cache = {}
        self.max_cache_size = 50000
        
        # Search statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
    def search(self, board: chess.Board, depth: int = 2, 
               time_limit_ms: float = 400.0, use_ml: bool = True) -> SearchResult:
        """
        Search for the best move using minimax with alpha-beta
        
        Args:
            board: Current board position
            depth: Search depth in plies
            time_limit_ms: Maximum time to search
            use_ml: Whether to use ML evaluation (if available)
            
        Returns:
            SearchResult with best move and analysis
        """
        self.nodes_searched = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.timer.start()
        
        # Clear cache periodically to prevent memory issues
        if len(self.tt_cache) > self.max_cache_size:
            self.tt_cache.clear()
        
        best_move = None
        best_score = float('-inf')
        pv = []
        
        # Check for immediate tactical threats first
        tactical_move = self._check_tactics(board)
        
        # Get legal moves and order them
        legal_moves = self._order_moves_enhanced(board, list(board.legal_moves))

        if not legal_moves:
            # No legal moves - checkmate or stalemate
            score = -1000.0 if board.is_check() else 0.0
            return SearchResult(None, score, self.nodes_searched, [], 
                              self.timer.elapsed_ms(), depth)

        # If we found a tactical move, prioritize it
        if tactical_move and tactical_move in legal_moves:
            legal_moves.remove(tactical_move)
            legal_moves.insert(0, tactical_move)

        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            # Check time limit
            if self.timer.is_time_up(time_limit_ms):
                break
                
            board.push(move)
            
            # Search this move with negamax
            score = -self._negamax(board, depth - 1, -beta, -alpha, use_ml, time_limit_ms)
            
            board.pop()
            
            if score > best_score:
                best_score = score
                best_move = move
                pv = [move]  # Simple PV for now

            alpha = max(alpha, score)
            if alpha >= beta:
                break  # Beta cutoff

        return SearchResult(best_move, best_score, self.nodes_searched, 
                          pv, self.timer.elapsed_ms(), depth)
    
    def _negamax(self, board: chess.Board, depth: int, alpha: float, 
                 beta: float, use_ml: bool, time_limit_ms: float) -> float:
        """
        Negamax search with alpha-beta pruning and transposition table
        """
        self.nodes_searched += 1
        
        # Check time limit
        if self.timer.is_time_up(time_limit_ms):
            return self._evaluate_position(board, use_ml)
        
        # Check transposition table
        fen = board.fen()
        if fen in self.tt_cache:
            cached_score, cached_depth = self.tt_cache[fen]
            if cached_depth >= depth:
                self.cache_hits += 1
                return cached_score
        
        self.cache_misses += 1
        
        # Terminal conditions
        if depth <= 0:
            score = self._evaluate_position(board, use_ml)
            self._store_in_cache(fen, score, depth)
            return score
        
        if board.is_checkmate():
            score = -9999.0 + (20 - depth)  # Prefer faster checkmates
            self._store_in_cache(fen, score, depth)
            return score
        
        if board.is_stalemate() or board.is_insufficient_material():
            self._store_in_cache(fen, 0.0, depth)
            return 0.0
        
        legal_moves = self._order_moves_enhanced(board, list(board.legal_moves))
        best_score = float('-inf')
        
        for move in legal_moves:
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha, use_ml, time_limit_ms)
            board.pop()
            
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            
            if alpha >= beta:
                break  # Beta cutoff
        
        self._store_in_cache(fen, best_score, depth)
        return best_score
    
    def _evaluate_position(self, board: chess.Board, use_ml: bool) -> float:
        """Evaluate position using ML or baseline evaluator"""
        if hasattr(self.evaluator, 'model') and use_ml and self.evaluator.model is not None:
            # Use ML evaluation
            return self.evaluator.evaluate(board)
        else:
            # Use baseline evaluation or fallback
            return self.evaluator.evaluate(board)
    
    def _check_tactics(self, board: chess.Board) -> Optional[chess.Move]:
        """Quick tactical check for obvious moves (mate in 1, free pieces)"""
        # Check for mate in 1
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            board.pop()
        
        # Look for free pieces (captures with no immediate recapture)
        best_capture = None
        best_value = 0
        
        piece_values = {
            chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9
        }
        
        for move in board.legal_moves:
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    value = piece_values.get(captured_piece.piece_type, 0)
                    
                    # Simple check: is the capturing piece defended?
                    board.push(move)
                    attackers = board.attackers(not board.turn, move.to_square)
                    board.pop()
                    
                    # If no immediate recapture, this is likely good
                    if not attackers and value > best_value:
                        best_capture = move
                        best_value = value
        
        return best_capture
    
    def _order_moves_enhanced(self, board: chess.Board, moves: List[chess.Move]) -> List[chess.Move]:
        """
        Enhanced move ordering for better alpha-beta pruning
        Priority: Mate threats > Captures (MVV-LVA) > Checks > Castling > Central moves
        """
        mate_threats = []
        captures = []
        checks = []
        castling = []
        others = []
        
        for move in moves:
            # Check for mate threats
            board.push(move)
            if board.is_checkmate():
                mate_threats.append(move)
                board.pop()
                continue
            board.pop()
            
            # Captures
            if board.is_capture(move):
                victim_value = self._get_capture_value(board, move)
                captures.append((move, victim_value))
            # Checks
            elif board.gives_check(move):
                checks.append(move)
            # Castling
            elif board.is_castling(move):
                castling.append(move)
            # Other moves
            else:
                # Prioritize central squares
                to_file = chess.square_file(move.to_square)
                to_rank = chess.square_rank(move.to_square)
                priority = 0
                if 2 <= to_file <= 5 and 2 <= to_rank <= 5:
                    priority = 1
                others.append((move, priority))
        
        # Sort captures by value (highest first)
        captures.sort(key=lambda x: x[1], reverse=True)
        others.sort(key=lambda x: x[1], reverse=True)
        
        # Return ordered moves
        return (mate_threats + 
                [move for move, _ in captures] + 
                checks + 
                castling + 
                [move for move, _ in others])
    
    def _get_capture_value(self, board: chess.Board, move: chess.Move) -> float:
        """Get the value of a capture for move ordering (MVV-LVA)"""
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None:
            return 0.0
        
        # Most Valuable Victim - Least Valuable Attacker
        victim_value = self._piece_value(captured_piece.piece_type)
        attacker_piece = board.piece_at(move.from_square)
        attacker_value = self._piece_value(attacker_piece.piece_type) if attacker_piece else 0
        
        return victim_value * 10 - attacker_value
    
    def _piece_value(self, piece_type: int) -> float:
        """Get piece value for MVV-LVA"""
        values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.2,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 100.0
        }
        return values.get(piece_type, 0.0)
    
    def _store_in_cache(self, fen: str, score: float, depth: int):
        """Store position in transposition table"""
        if len(self.tt_cache) < self.max_cache_size:
            self.tt_cache[fen] = (score, depth)
    
    def clear_cache(self):
        """Clear transposition table"""
        self.tt_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> dict:
        """Get cache performance statistics"""
        total_lookups = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_lookups if total_lookups > 0 else 0
        
        return {
            'cache_size': len(self.tt_cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate
        }

# Legacy compatibility - maintain old interface
def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool, evaluator) -> float:
    """Legacy minimax function for backward compatibility"""
    search = ChessSearch(evaluator)
    # Convert to negamax call
    if maximizing:
        return search._negamax(board, depth, alpha, beta, True, 1000)
    else:
        return -search._negamax(board, depth, -beta, -alpha, True, 1000)