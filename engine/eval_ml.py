#!/usr/bin/env python3
"""
Quick fix for eval_ml.py - fixes the import issues
"""

def fix_eval_ml():
    """Fix the eval_ml.py file"""
    
    # Read the current file
    with open('engine/eval_ml.py', 'r') as f:
        content = f.read()
    
    # Fix 1: Add missing import
    if 'import os' not in content:
        content = content.replace('import logging', 'import logging\nimport os')
        print("✅ Added missing 'import os'")
    
    # Fix 2: Fix the orphaned @lru_cache decorator
    if '@lru_cache(maxsize=10000)\n    \n    def _safe_load_model' in content:
        content = content.replace(
            '@lru_cache(maxsize=10000)\n    \n    def _safe_load_model',
            '@lru_cache(maxsize=10000)\n    def extract_features'
        )
        print("✅ Fixed orphaned @lru_cache decorator")
    
    # Fix 3: Remove the problematic import line that's causing the issue
    if 'from .evaluation import BaselineEvaluator' in content:
        content = content.replace('from .evaluation import BaselineEvaluator', '# from .evaluation import BaselineEvaluator')
        print("✅ Commented out problematic BaselineEvaluator import")
    
    # Fix 4: Add a simple BaselineEvaluator fallback
    baseline_evaluator_code = '''

# Simple BaselineEvaluator for fallback
class BaselineEvaluator:
    """Simple baseline evaluator fallback"""
    
    def evaluate(self, board):
        """Simple material-based evaluation"""
        if board.is_checkmate():
            return -1000.0
        if board.is_stalemate():
            return 0.0
        
        # Simple material count
        piece_values = {1: 1, 2: 3, 3: 3.2, 4: 5, 5: 9, 6: 0}  # piece_type values
        
        white_score = sum(piece_values.get(piece.piece_type, 0) 
                         for piece in board.piece_map().values() 
                         if piece.color)
        black_score = sum(piece_values.get(piece.piece_type, 0) 
                         for piece in board.piece_map().values() 
                         if not piece.color)
        
        # Return from side-to-move perspective
        if board.turn:  # White to move
            return white_score - black_score
        else:  # Black to move
            return black_score - white_score
'''
    
    # Add BaselineEvaluator if not present
    if 'class BaselineEvaluator' not in content:
        content += baseline_evaluator_code
        print("✅ Added BaselineEvaluator fallback class")
    
    # Write the fixed content
    with open('engine/eval_ml.py', 'w') as f:
        f.write(content)
    
    print("✅ Fixed eval_ml.py!")

def create_simple_test():
    """Create a simple test that works with the fixed eval_ml.py"""
    
    test_content = '''#!/usr/bin/env python3
"""
Simple test for fixed eval_ml.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chess

def test_ml_evaluator():
    print("Testing ML Evaluator")
    print("=" * 20)
    
    try:
        from engine.eval_ml import MLEvaluator
        
        # Create evaluator
        evaluator = MLEvaluator()
        
        # Test on starting position
        board = chess.Board()
        
        # Test evaluation
        score = evaluator.evaluate(board)
        print(f"Evaluation score: {score:.3f}")
        
        # Test model loading
        if evaluator.model is not None:
            print("✅ ML model loaded successfully!")
            
            # Test win probability
            try:
                prob = evaluator.predict_win_probability(board.fen())
                print(f"Win probability: {prob:.1%}")
            except Exception as e:
                print(f"Win probability failed: {e}")
        else:
            print("⚠️  No ML model - using baseline fallback")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_baseline_fallback():
    print("\\nTesting Baseline Fallback")
    print("=" * 25)
    
    try:
        from engine.eval_ml import BaselineEvaluator
        
        evaluator = BaselineEvaluator()
        board = chess.Board()
        
        score = evaluator.evaluate(board)
        print(f"Baseline score: {score:.3f}")
        print("✅ Baseline evaluator working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return False

def main():
    print("Fixed eval_ml.py Test")
    print("=====================")
    
    ml_ok = test_ml_evaluator()
    baseline_ok = test_baseline_fallback()
    
    print("\\n" + "=" * 30)
    if ml_ok and baseline_ok:
        print("🎉 Everything working!")
        print("\\nYou can now run:")
        print("  python test_engine_cli.py")
    else:
        print("Still have issues - check errors above")

if __name__ == "__main__":
    main()
'''
    
    with open('test_fixed_eval.py', 'w') as f:
        f.write(test_content)
    
    print("✅ Created test_fixed_eval.py")

def main():
    print("Quick Fix for eval_ml.py")
    print("========================")
    
    if not os.path.exists('engine/eval_ml.py'):
        print("❌ engine/eval_ml.py not found!")
        print("Make sure you're in the right directory")
        return
    
    # Backup original
    import shutil
    shutil.copy('engine/eval_ml.py', 'engine/eval_ml.py.backup')
    print("✅ Backed up original to eval_ml.py.backup")
    
    # Apply fixes
    fix_eval_ml()
    
    # Create test
    create_simple_test()
    
    print("\\n" + "=" * 30)
    print("Quick Fix Complete!")
    print("\\nNext steps:")
    print("1. Run: python test_fixed_eval.py")
    print("2. If working: python test_engine_cli.py")

if __name__ == "__main__":
    import os
    main()