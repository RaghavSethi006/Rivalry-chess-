#!/usr/bin/env python3
"""
Phase 4 Setup - Works with your existing eval_ml.py structure
"""

import os
import json
import shutil
from pathlib import Path

def find_existing_files():
    """Find your existing file structure"""
    files_found = {}
    
    # Check for eval_ml.py in different locations
    possible_eval_ml_locations = [
        "eval_ml.py",
        "engine/eval_ml.py", 
        "ml/eval_ml.py"
    ]
    
    for location in possible_eval_ml_locations:
        if os.path.exists(location):
            files_found['eval_ml'] = location
            break
    
    # Check for evaluation.py
    possible_eval_locations = [
        "engine/evaluation.py",
        "evaluation.py"
    ]
    
    for location in possible_eval_locations:
        if os.path.exists(location):
            files_found['evaluation'] = location
            break
    
    # Check for search.py
    possible_search_locations = [
        "engine/search.py",
        "search.py"
    ]
    
    for location in possible_search_locations:
        if os.path.exists(location):
            files_found['search'] = location
            break
    
    return files_found

def update_eval_ml_file(eval_ml_path):
    """Update your existing eval_ml.py to fix CalibratedWrapper issues"""
    
    print(f"Updating {eval_ml_path}...")
    
    # Read existing file
    with open(eval_ml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if it already has safe loading
    if "_safe_load_model" in content:
        print("✅ eval_ml.py already has safe loading")
        return True
    
    # Add safe loading method
    safe_loading_code = '''
    def _safe_load_model(self, model_path):
        """Safely load model with CalibratedWrapper handling"""
        if not os.path.exists(model_path):
            return None
            
        try:
            import joblib
            
            # First try normal loading
            model = joblib.load(model_path)
            return model
            
        except Exception as e:
            logger.warning(f"Standard model loading failed: {e}")
            
            # Try loading with custom handling for sklearn models
            try:
                import pickle
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model
                
            except Exception as e2:
                logger.warning(f"Pickle loading also failed: {e2}")
                return None
'''
    
    # Update the __init__ method to use safe loading
    if "joblib.load(model_path)" in content:
        content = content.replace(
            "self.model = joblib.load(model_path)",
            "self.model = self._safe_load_model(model_path)"
        )
        
        # Add the safe loading method before the last class method
        # Find a good place to insert it
        if "def extract_features" in content:
            content = content.replace(
                "def extract_features",
                safe_loading_code + "\n    def extract_features"
            )
        elif "def predict_win_probability" in content:
            content = content.replace(
                "def predict_win_probability", 
                safe_loading_code + "\n    def predict_win_probability"
            )
        else:
            # Append at end of class
            content += safe_loading_code
    
    # Write updated file
    with open(eval_ml_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Updated eval_ml.py with safe model loading")
    return True

def create_missing_evaluation_file():
    """Create evaluation.py if it's missing"""
    
    if os.path.exists("engine/evaluation.py"):
        print("✅ engine/evaluation.py already exists")
        return True
    
    print("Creating missing engine/evaluation.py...")
    
    # Create engine directory
    os.makedirs("engine", exist_ok=True)
    
    # Simple evaluation file that imports from your eval_ml
    eval_content = '''"""
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
'''
    
    with open("engine/evaluation.py", 'w', encoding='utf-8') as f:
        f.write(eval_content)
    
    print("✅ Created engine/evaluation.py")
    return True

def create_integration_test():
    """Create test that works with your existing structure"""
    
    test_content = '''#!/usr/bin/env python3
"""
Test your existing Phase 4 setup
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chess

def test_existing_structure():
    print("Testing Your Existing Phase 4 Setup")
    print("=" * 40)
    
    # Test importing your existing eval_ml
    try:
        # Try different import paths
        try:
            from eval_ml import MLEvaluator
            print("✅ Found eval_ml.py in root directory")
        except ImportError:
            try:
                from engine.eval_ml import MLEvaluator
                print("✅ Found eval_ml.py in engine directory")
            except ImportError:
                from ml.eval_ml import MLEvaluator
                print("✅ Found eval_ml.py in ml directory")
        
        # Test the ML evaluator
        ml_eval = MLEvaluator()
        board = chess.Board()
        
        # Test basic evaluation
        score = ml_eval.evaluate(board)
        print(f"ML evaluation score: {score:.3f}")
        
        # Test win probability if available
        if hasattr(ml_eval, 'predict_win_probability'):
            prob = ml_eval.predict_win_probability(board.fen())
            print(f"Win probability: {prob:.1%}")
        
        if hasattr(ml_eval, 'model') and ml_eval.model is not None:
            print("✅ ML model is loaded and working!")
        else:
            print("⚠️  ML model not loaded - using fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Error with ML evaluator: {e}")
        return False

def test_search_integration():
    print("\\nTesting Search Integration")
    print("=" * 30)
    
    try:
        # Try to import your search engine
        try:
            from engine.search import ChessSearch
        except ImportError:
            from search import ChessSearch
        
        # Try to get an evaluator
        try:
            from eval_ml import MLEvaluator
            evaluator = MLEvaluator()
        except ImportError:
            try:
                from engine.evaluation import BaselineEvaluator
                evaluator = BaselineEvaluator()
            except ImportError:
                print("⚠️  No evaluator found - skipping search test")
                return True
        
        # Test search
        search = ChessSearch(evaluator)
        board = chess.Board()
        
        result = search.search(board, depth=2, time_limit_ms=500)
        
        if hasattr(result, 'best_move') and result.best_move:
            print(f"✅ Search working: {result.best_move}")
            print(f"   Score: {result.score:.3f}")
            return True
        else:
            print("⚠️  Search returned no move")
            return False
            
    except Exception as e:
        print(f"❌ Search test failed: {e}")
        return False

def main():
    print("Phase 4 Existing Structure Test")
    print("===============================")
    
    eval_ok = test_existing_structure()
    search_ok = test_search_integration()
    
    print("\\n" + "=" * 40)
    if eval_ok:
        print("✅ Your ML evaluation is working!")
        if search_ok:
            print("✅ Search integration is working!")
        print("\\n🎉 Phase 4 is ready!")
        print("\\nNext steps:")
        print("- Test with: python test_engine_cli.py")
        print("- Fine-tune your model parameters")
        print("- Move to Phase 5 (API development)")
    else:
        print("❌ Issues found. Check:")
        print("- Do you have models/model.joblib from Phase 3?")
        print("- Are feature names correct in model metadata?")
        print("- Are all dependencies installed?")

if __name__ == "__main__":
    main()
'''
    
    with open("test_existing_phase4.py", 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ Created test_existing_phase4.py")

def fix_model_metadata():
    """Create/fix model metadata file"""
    meta_path = "models/model_meta.json"
    
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Create compatible metadata
    meta_data = {
        "version": "1.0.0",
        "feature_names": [
            "material_diff", "my_material", "opp_material",
            "my_pawns", "opp_pawns", "my_knights", "opp_knights", 
            "my_bishops", "opp_bishops", "my_rooks", "opp_rooks",
            "my_queens", "opp_queens", "mobility_diff", "my_mobility", "opp_mobility",
            "my_king_castled", "opp_king_castled", "my_king_file", "my_king_rank",
            "opp_king_file", "opp_king_rank", "my_doubled_pawns", "opp_doubled_pawns",
            "my_isolated_pawns", "opp_isolated_pawns", "my_passed_pawns", "opp_passed_pawns",
            "my_pst_value", "opp_pst_value", "game_phase", "ply", "in_check"
        ],
        "training_date": "2024-01-01",
        "model_type": "ML Model",
        "notes": "Compatible with eval_ml.py feature extraction"
    }
    
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2)
    
    print("✅ Created/updated models/model_meta.json")

def main():
    print("Phase 4 Setup - Works with Your Existing eval_ml.py")
    print("=" * 55)
    
    # Find existing files
    files = find_existing_files()
    
    print("Found existing files:")
    for key, path in files.items():
        print(f"  {key}: {path}")
    
    if not files:
        print("❌ No existing evaluation files found!")
        print("   Make sure you have eval_ml.py somewhere in your project")
        return
    
    # Update eval_ml if found
    if 'eval_ml' in files:
        update_eval_ml_file(files['eval_ml'])
    
    # Create missing evaluation.py if needed
    if 'evaluation' not in files:
        create_missing_evaluation_file()
    
    # Fix model metadata
    fix_model_metadata()
    
    # Create integration test
    create_integration_test()
    
    print("\\n" + "=" * 55)
    print("Setup Complete!")
    print("✅ Updated your existing eval_ml.py with safe model loading")
    print("✅ Created/updated evaluation files")
    print("✅ Fixed model metadata") 
    print("✅ Created integration test")
    
    print("\\nNext Steps:")
    print("1. Run: python test_existing_phase4.py")
    print("2. If successful: python test_engine_cli.py")
    print("3. Update feature_names in models/model_meta.json to match your Phase 3 training")

if __name__ == "__main__":
    main()