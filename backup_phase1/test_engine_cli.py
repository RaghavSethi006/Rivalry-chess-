#!/usr/bin/env python3
"""
CLI test script for the chess engine - Phase 4 enhanced version
Run this to test your upgraded ML-powered engine interactively
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import chess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_phase_4_availability():
    """Test if Phase 4 ML components are available"""
    try:
        from engine.evaluation import MLEvaluator
        ml_eval = MLEvaluator()
        
        if ml_eval.model is not None:
            print("✅ Phase 4 ML evaluation is available!")
            print(f"   Model version: {ml_eval.version}")
            print(f"   Features: {len(ml_eval.feature_names)}")
            return True, ml_eval
        else:
            print("⚠️  Phase 4 ML model not found - using baseline evaluation")
            return False, ml_eval
            
    except Exception as e:
        print(f"❌ Phase 4 components not available: {e}")
        print("   Falling back to Phase 1 engine...")
        return False, None

def print_board(fen: str):
    """Print the chess board"""
    board = chess.Board(fen)
    print(board)
    print(f"FEN: {fen}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print()

def analyze_position_detailed(evaluator, fen: str):
    """Provide detailed position analysis"""
    print("\n🔍 Detailed Position Analysis")
    print("=" * 40)
    
    board = chess.Board(fen)
    
    # Basic evaluation
    score = evaluator.evaluate(board)
    print(f"Position score: {score:.3f}")
    
    # If ML evaluator, show more details
    if hasattr(evaluator, 'model') and evaluator.model is not None:
        try:
            win_prob = evaluator.predict_win_probability(fen)
            print(f"Win probability: {win_prob:.1%}")
            print(f"ML score: {evaluator.probability_to_score(win_prob):.3f}")
        except Exception as e:
            print(f"ML analysis failed: {e}")
    
    # Game status
    if board.is_check():
        print("⚠️  IN CHECK")
    if board.is_checkmate():
        print("🎯 CHECKMATE")
    if board.is_stalemate():
        print("🤝 STALEMATE")
    
    # Legal moves count
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    print("=" * 40)

def main():
    """Interactive CLI test"""
    print("🏁 Chess Engine Phase 4 - Enhanced CLI Test")
    print("============================================")
    
    # Test Phase 4 availability
    has_ml, ml_evaluator = test_phase_4_availability()
    
    # Try to import the engine
    try:
        if has_ml:
            # Use enhanced Phase 4 components
            from engine.evaluation import MLEvaluator
            from engine.search import ChessSearch
            
            evaluator = MLEvaluator()
            search_engine = ChessSearch(evaluator)
            print("🚀 Using Phase 4 enhanced engine with ML evaluation")
        else:
            # Fallback to Phase 1 engine
            from engine import default_engine
            print("📚 Using Phase 1 baseline engine")
            
    except ImportError as e:
        print(f"❌ Could not import engine: {e}")
        return
    
    # Starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    # Configuration
    search_depth = 2
    time_limit = 500
    use_ml_eval = has_ml and ml_evaluator and ml_evaluator.model is not None
    
    print(f"\n⚙️  Engine Configuration:")
    print(f"   Search depth: {search_depth}")
    print(f"   Time limit: {time_limit}ms")
    print(f"   ML evaluation: {'ON' if use_ml_eval else 'OFF'}")
    
    while True:
        print_board(fen)
        
        board = chess.Board(fen)
        if board.is_game_over():
            if board.is_checkmate():
                winner = "Black" if board.turn else "White"
                print(f"🏆 Checkmate! {winner} wins!")
            else:
                print("🤝 Game over - Draw!")
            break
        
        print("\nOptions:")
        print("1. Get engine move (auto-play engine)")
        print("2. Make your move (enter UCI format like 'e2e4')")
        print("3. Analyze position (detailed analysis)")
        print("4. Set position (enter FEN)")
        print("5. Configure engine (depth, time, ML)")
        print("6. Benchmark search")
        print("7. Quit")
        
        choice = input("\nChoice (1-7): ").strip()
        
        if choice == '1':
            # Get engine move
            print("🤖 Engine thinking...")
            
            if has_ml:
                # Use Phase 4 engine
                result = search_engine.search(
                    board, 
                    depth=search_depth, 
                    time_limit_ms=time_limit,
                    use_ml=use_ml_eval
                )
                
                if result.best_move:
                    print(f"✅ Engine move: {result.best_move}")
                    print(f"   Score: {result.score:.3f}")
                    print(f"   Nodes: {result.nodes_searched}")
                    print(f"   Time: {result.duration_ms:.1f}ms")
                    print(f"   Depth: {result.depth_reached}")
                    
                    # Cache stats if available
                    if hasattr(search_engine, 'get_cache_stats'):
                        cache_stats = search_engine.get_cache_stats()
                        print(f"   Cache hits: {cache_stats['cache_hits']}/{cache_stats['cache_hits'] + cache_stats['cache_misses']} ({cache_stats['hit_rate']:.1%})")
                    
                    # Make the move
                    board.push(result.best_move)
                    fen = board.fen()
                    
                    if board.is_check():
                        print("👑 Check!")
                else:
                    print("❌ Engine couldn't find a move")
                    
            else:
                # Use Phase 1 engine
                result = default_engine.get_best_move(fen, depth=search_depth, time_limit_ms=time_limit)
                
                if result.get('error'):
                    print(f"❌ Engine error: {result['error']}")
                    continue
                
                best_move = result.get('bestMove')
                if not best_move:
                    print("❌ Engine couldn't find a move")
                    continue
                
                print(f"✅ Engine move: {best_move}")
                print(f"   Score: {result.get('score', 0):.3f}")
                print(f"   Nodes: {result.get('nodes', 0)}")
                print(f"   Time: {result.get('duration_ms', 0):.1f}ms")
                
                # Make the move
                move_result = default_engine.make_move(fen, best_move)
                if move_result.get('error'):
                    print(f"❌ Move error: {move_result['error']}")
                    continue
                
                fen = move_result['fen']
                
                if move_result.get('check'):
                    print("👑 Check!")
            
        elif choice == '2':
            # Player move
            move_uci = input("Enter your move (UCI format, e.g., e2e4): ").strip()
            
            if not move_uci:
                continue
            
            try:
                move = chess.Move.from_uci(move_uci)
                if move in board.legal_moves:
                    board.push(move)
                    fen = board.fen()
                    print(f"✅ Move made: {move_uci}")
                    
                    if board.is_check():
                        print("👑 Check!")
                else:
                    print(f"❌ Illegal move: {move_uci}")
            except ValueError:
                print(f"❌ Invalid move format: {move_uci}")
                
        elif choice == '3':
            # Detailed position analysis
            if has_ml and ml_evaluator:
                analyze_position_detailed(ml_evaluator, fen)
            else:
                # Basic analysis for Phase 1
                eval_result = default_engine.evaluate_position(fen)
                print(f"\n📊 Position evaluation: {eval_result.get('score', 'Error')}")
                
        elif choice == '4':
            # Set custom position
            new_fen = input("Enter FEN string: ").strip()
            try:
                # Validate FEN
                chess.Board(new_fen)
                fen = new_fen
                print("✅ Position set!")
            except ValueError:
                print("❌ Invalid FEN string")
                
        elif choice == '5':
            # Configure engine
            print(f"\nCurrent configuration:")
            print(f"  Depth: {search_depth}")
            print(f"  Time limit: {time_limit}ms")
            print(f"  ML evaluation: {'ON' if use_ml_eval else 'OFF'}")
            
            new_depth = input(f"New depth (current {search_depth}): ").strip()
            if new_depth.isdigit():
                search_depth = int(new_depth)
                print(f"✅ Depth set to {search_depth}")
            
            new_time = input(f"New time limit in ms (current {time_limit}): ").strip()
            if new_time.isdigit():
                time_limit = int(new_time)
                print(f"✅ Time limit set to {time_limit}ms")
                
            if has_ml and ml_evaluator and ml_evaluator.model is not None:
                ml_choice = input("Use ML evaluation? (y/n): ").strip().lower()
                use_ml_eval = ml_choice in ['y', 'yes', '1']
                print(f"✅ ML evaluation {'enabled' if use_ml_eval else 'disabled'}")
                
        elif choice == '6':
            # Benchmark search
            print("\n⏱️  Running benchmark...")
            if has_ml:
                import time
                start_time = time.time()
                
                result = search_engine.search(
                    board, 
                    depth=3,  # Higher depth for benchmark
                    time_limit_ms=2000,
                    use_ml=use_ml_eval
                )
                
                duration = time.time() - start_time
                nps = result.nodes_searched / duration if duration > 0 else 0
                
                print(f"📈 Benchmark results:")
                print(f"   Depth: 3")
                print(f"   Nodes: {result.nodes_searched}")
                print(f"   Time: {duration:.3f}s")
                print(f"   NPS: {nps:.0f} nodes/second")
                
                if hasattr(search_engine, 'get_cache_stats'):
                    cache_stats = search_engine.get_cache_stats()
                    print(f"   Cache efficiency: {cache_stats['hit_rate']:.1%}")
            else:
                print("Benchmark only available with Phase 4 engine")
                
        elif choice == '7':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")
        
        print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()