"""
PGN to rows converter - extracts positions and game metadata from PGN files.
"""
import chess
import chess.pgn
import io
import logging
from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GamePosition:
    """Represents a single position from a game."""
    fen: str
    side_to_move: bool  # True = White, False = Black
    result: float  # 1.0 = White wins, 0.5 = Draw, 0.0 = Black wins
    ply: int
    game_id: str
    white_elo: Optional[int] = None
    black_elo: Optional[int] = None
    time_control: Optional[str] = None
    is_rated: bool = True

class PGNProcessor:
    """Processes PGN files and extracts training positions."""
    
    def __init__(self, 
                 sample_every_n_plies: int = 2,
                 min_game_length: int = 10,
                 max_game_length: int = 200,
                 min_elo: int = 1000,
                 max_elo: int = 2500):
        """
        Args:
            sample_every_n_plies: Sample positions every N plies to reduce correlation
            min_game_length: Skip games shorter than this
            max_game_length: Skip games longer than this (likely analysis)
            min_elo: Skip games with players below this rating
            max_elo: Skip games with players above this rating (too strong for our target)
        """
        self.sample_every_n_plies = sample_every_n_plies
        self.min_game_length = min_game_length
        self.max_game_length = max_game_length
        self.min_elo = min_elo
        self.max_elo = max_elo
        
    def parse_result(self, result_str: str) -> Optional[float]:
        """Parse PGN result string to numeric value."""
        result_map = {
            "1-0": 1.0,
            "0-1": 0.0,
            "1/2-1/2": 0.5,
            "*": None  # Unfinished game
        }
        return result_map.get(result_str)
    
    def extract_elo(self, headers: Dict[str, str]) -> Tuple[Optional[int], Optional[int]]:
        """Extract ELO ratings from game headers."""
        try:
            white_elo = int(headers.get("WhiteElo", "0")) if headers.get("WhiteElo", "0") != "?" else None
            black_elo = int(headers.get("BlackElo", "0")) if headers.get("BlackElo", "0") != "?" else None
            return white_elo, black_elo
        except (ValueError, TypeError):
            return None, None
    
    def should_skip_game(self, headers: Dict[str, str], moves_count: int) -> Tuple[bool, str]:
        """Determine if game should be skipped based on criteria."""
        
        # Check game length
        if moves_count < self.min_game_length:
            return True, f"too_short_{moves_count}"
        if moves_count > self.max_game_length:
            return True, f"too_long_{moves_count}"
        
        # Check result
        result = self.parse_result(headers.get("Result", "*"))
        if result is None:
            return True, "unfinished"
        
        # Check ELO ratings
        white_elo, black_elo = self.extract_elo(headers)
        if white_elo and (white_elo < self.min_elo or white_elo > self.max_elo):
            return True, f"white_elo_{white_elo}"
        if black_elo and (black_elo < self.min_elo or black_elo > self.max_elo):
            return True, f"black_elo_{black_elo}"
        
        # Skip if both ELOs are missing
        if not white_elo and not black_elo:
            return True, "no_elos"
            
        return False, "accepted"
    
    def process_game(self, game: chess.pgn.Game, game_id: str) -> List[GamePosition]:
        """Extract positions from a single game."""
        headers = dict(game.headers)
        
        # Get basic game info
        result = self.parse_result(headers.get("Result", "*"))
        if result is None:
            return []
            
        white_elo, black_elo = self.extract_elo(headers)
        time_control = headers.get("TimeControl")
        is_rated = headers.get("Event", "").lower().find("rated") != -1
        
        # Play through the game and sample positions
        board = game.board()
        positions = []
        ply = 0
        
        for move in game.mainline_moves():
            ply += 1
            
            # Sample position before making the move
            if ply % self.sample_every_n_plies == 0:
                # Skip very early positions (opening book territory)
                if ply >= 6:
                    position = GamePosition(
                        fen=board.fen(),
                        side_to_move=board.turn,  # True = White's turn
                        result=result,
                        ply=ply,
                        game_id=game_id,
                        white_elo=white_elo,
                        black_elo=black_elo,
                        time_control=time_control,
                        is_rated=is_rated
                    )
                    positions.append(position)
            
            # Make the move
            try:
                board.push(move)
            except ValueError as e:
                logger.warning(f"Invalid move in game {game_id}: {e}")
                break
                
        return positions
    
    def process_pgn_file(self, pgn_path: str, max_games: Optional[int] = None) -> Iterator[GamePosition]:
        """Process a PGN file and yield positions."""
        games_processed = 0
        games_skipped = 0
        skip_reasons = {}
        positions_yielded = 0
        
        logger.info(f"Processing PGN file: {pgn_path}")
        
        with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    games_processed += 1
                    game_id = f"g{games_processed}"
                    
                    if max_games and games_processed > max_games:
                        break
                    
                    # Quick game validation
                    moves_list = list(game.mainline_moves())
                    should_skip, reason = self.should_skip_game(dict(game.headers), len(moves_list))
                    
                    if should_skip:
                        games_skipped += 1
                        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                        continue
                    
                    # Extract positions from this game
                    positions = self.process_game(game, game_id)
                    
                    for position in positions:
                        yield position
                        positions_yielded += 1
                    
                    if games_processed % 1000 == 0:
                        logger.info(f"Processed {games_processed} games, "
                                  f"skipped {games_skipped}, "
                                  f"yielded {positions_yielded} positions")
                        
                except Exception as e:
                    logger.warning(f"Error processing game {games_processed}: {e}")
                    continue
        
        logger.info(f"Finished processing {pgn_path}")
        logger.info(f"Total games processed: {games_processed}")
        logger.info(f"Total games skipped: {games_skipped}")
        logger.info(f"Skip reasons: {skip_reasons}")
        logger.info(f"Total positions yielded: {positions_yielded}")

def test_pgn_processing():
    """Test the PGN processor with a simple example."""
    # Create a simple test PGN
    test_pgn = """
[Event "Test Game"]
[Site "Test"]
[Date "2023.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]
[WhiteElo "1500"]
[BlackElo "1400"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 1-0
"""
    
    # Write test PGN to file
    with open("test_game.pgn", "w") as f:
        f.write(test_pgn)
    
    # Process it
    processor = PGNProcessor(sample_every_n_plies=4)  # Sample less frequently for test
    positions = list(processor.process_pgn_file("test_game.pgn"))
    
    print(f"Extracted {len(positions)} positions:")
    for i, pos in enumerate(positions):
        print(f"{i+1}. Ply {pos.ply}, {'White' if pos.side_to_move else 'Black'} to move")
        print(f"   FEN: {pos.fen}")
        print(f"   Result: {pos.result} (White wins)")
        print()

if __name__ == "__main__":
    test_pgn_processing()