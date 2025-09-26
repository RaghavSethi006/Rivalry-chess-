"""
Dataset creation pipeline - converts PGN games to feature vectors with labels.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional, Tuple
import argparse
import json
from sklearn.model_selection import train_test_split
import joblib

from pgn_to_rows import PGNProcessor, GamePosition
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engine.features import ChessFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChessDatasetBuilder:
    """Builds training datasets from PGN files."""
    
    def __init__(self, 
                 data_dir: str = "data",
                 sample_every_n_plies: int = 2,
                 min_elo: int = 1000,
                 max_elo: int = 2200,
                 balance_results: bool = True,
                 max_positions_per_result: int = 50000):
        """
        Args:
            data_dir: Base directory for data storage
            sample_every_n_plies: Sample positions every N plies
            min_elo: Minimum ELO rating to include
            max_elo: Maximum ELO rating to include
            balance_results: Whether to balance win/draw/loss samples
            max_positions_per_result: Max positions per result class (for balancing)
        """
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.pgn_processor = PGNProcessor(
            sample_every_n_plies=sample_every_n_plies,
            min_elo=min_elo,
            max_elo=max_elo
        )
        self.feature_extractor = ChessFeatureExtractor()
        self.balance_results = balance_results
        self.max_positions_per_result = max_positions_per_result
        
    def process_pgn_files(self, pgn_files: List[str], max_games_per_file: Optional[int] = None) -> List[GamePosition]:
        """Process multiple PGN files and extract positions."""
        all_positions = []
        
        for pgn_file in pgn_files:
            logger.info(f"Processing {pgn_file}")
            pgn_path = self.raw_dir / pgn_file
            
            if not pgn_path.exists():
                logger.error(f"PGN file not found: {pgn_path}")
                continue
                
            positions = list(self.pgn_processor.process_pgn_file(str(pgn_path), max_games_per_file))
            all_positions.extend(positions)
            logger.info(f"Extracted {len(positions)} positions from {pgn_file}")
        
        logger.info(f"Total positions extracted: {len(all_positions)}")
        return all_positions
    
    def positions_to_dataframe(self, positions: List[GamePosition]) -> pd.DataFrame:
        """Convert GamePosition objects to DataFrame with features and labels."""
        logger.info("Converting positions to features...")
        
        rows = []
        feature_extraction_errors = 0
        
        for i, pos in enumerate(positions):
            try:
                # Extract features
                features = self.feature_extractor.extract_features_from_fen(pos.fen, pos.ply)
                
                # Calculate label from game result and side to move
                if pos.side_to_move:  # White to move
                    label = pos.result  # 1.0=white wins, 0.5=draw, 0.0=black wins
                else:  # Black to move
                    label = 1.0 - pos.result  # Flip perspective
                
                # Create row
                row = {
                    'fen': pos.fen,
                    'label': label,
                    'side_to_move': pos.side_to_move,
                    'game_result': pos.result,
                    'ply': pos.ply,
                    'game_id': pos.game_id,
                    'white_elo': pos.white_elo,
                    'black_elo': pos.black_elo,
                    'avg_elo': (pos.white_elo + pos.black_elo) / 2 if pos.white_elo and pos.black_elo else None,
                    'is_rated': pos.is_rated
                }
                
                # Add all features
                row.update(features)
                rows.append(row)
                
            except Exception as e:
                feature_extraction_errors += 1
                if feature_extraction_errors <= 10:  # Only log first few errors
                    logger.warning(f"Feature extraction failed for position {i}: {e}")
                continue
            
            if (i + 1) % 10000 == 0:
                logger.info(f"Processed {i + 1}/{len(positions)} positions")
        
        if feature_extraction_errors > 0:
            logger.warning(f"Feature extraction failed for {feature_extraction_errors} positions")
        
        df = pd.DataFrame(rows)
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        return df
    
    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance the dataset by result classes."""
        if not self.balance_results:
            return df
        
        logger.info("Balancing dataset by result classes...")
        
        # Define result classes with some tolerance for floating point
        win_mask = df['label'] > 0.75  # Close to 1.0
        draw_mask = (df['label'] >= 0.25) & (df['label'] <= 0.75)  # Close to 0.5
        loss_mask = df['label'] < 0.25  # Close to 0.0
        
        win_positions = df[win_mask]
        draw_positions = df[draw_mask]
        loss_positions = df[loss_mask]
        
        logger.info(f"Before balancing - Wins: {len(win_positions)}, Draws: {len(draw_positions)}, Losses: {len(loss_positions)}")
        
        # Sample up to max_positions_per_result from each class
        sampled_dfs = []
        for positions, class_name in [(win_positions, 'wins'), (draw_positions, 'draws'), (loss_positions, 'losses')]:
            if len(positions) > self.max_positions_per_result:
                sampled = positions.sample(n=self.max_positions_per_result, random_state=42)
                logger.info(f"Sampled {len(sampled)} {class_name} from {len(positions)}")
            else:
                sampled = positions
                logger.info(f"Using all {len(sampled)} {class_name}")
            sampled_dfs.append(sampled)
        
        balanced_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"After balancing: {len(balanced_df)} total positions")
        return balanced_df
    
    def split_dataset(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/val/test, ensuring no game leakage."""
        logger.info("Splitting dataset by games to avoid leakage...")
        
        # Get unique games and their stats
        game_stats = df.groupby('game_id').agg({
            'label': ['count', 'mean'],
            'avg_elo': 'first',
            'game_result': 'first'
        }).reset_index()
        
        game_stats.columns = ['game_id', 'position_count', 'avg_label', 'avg_elo', 'game_result']
        
        logger.info(f"Total games: {len(game_stats)}")
        
        # Check if we have enough games for stratified splitting
        result_counts = game_stats['game_result'].value_counts()
        min_class_size = result_counts.min()
        
        if min_class_size < 3:
            logger.warning(f"Not enough games for stratified split (min class has {min_class_size} games). Using random split.")
            # Simple random split without stratification
            train_games, temp_games = train_test_split(
                game_stats, 
                test_size=test_size + val_size, 
                random_state=42
            )
            
            if len(temp_games) >= 2:
                val_games, test_games = train_test_split(
                    temp_games,
                    test_size=test_size / (test_size + val_size),
                    random_state=42
                )
            else:
                # If very few games, put all temp games in test
                val_games = temp_games.head(0)  # Empty dataframe
                test_games = temp_games
                
        else:
            # Stratified split
            train_games, temp_games = train_test_split(
                game_stats, 
                test_size=test_size + val_size, 
                random_state=42,
                stratify=game_stats['game_result']
            )
            
            # Check if temp_games has enough for stratified val/test split
            temp_result_counts = temp_games['game_result'].value_counts()
            if temp_result_counts.min() >= 2:
                val_games, test_games = train_test_split(
                    temp_games,
                    test_size=test_size / (test_size + val_size),
                    random_state=42,
                    stratify=temp_games['game_result']
                )
            else:
                # Random split for val/test
                val_games, test_games = train_test_split(
                    temp_games,
                    test_size=test_size / (test_size + val_size),
                    random_state=42
                )
        
        logger.info(f"Games split - Train: {len(train_games)}, Val: {len(val_games)}, Test: {len(test_games)}")
        
        # Create position splits based on game splits
        train_df = df[df['game_id'].isin(train_games['game_id'])].reset_index(drop=True)
        val_df = df[df['game_id'].isin(val_games['game_id'])].reset_index(drop=True)
        test_df = df[df['game_id'].isin(test_games['game_id'])].reset_index(drop=True)
        
        logger.info(f"Positions split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_dataset(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save datasets and metadata."""
        logger.info("Saving datasets...")
        
        # Save DataFrames
        train_df.to_parquet(self.processed_dir / "train.parquet", index=False)
        val_df.to_parquet(self.processed_dir / "val.parquet", index=False)
        test_df.to_parquet(self.processed_dir / "test.parquet", index=False)
        
        # Save combined dataset for convenience
        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        full_df.to_parquet(self.processed_dir / "full_dataset.parquet", index=False)
        
        # Save feature names and metadata
        metadata = {
            'feature_names': self.feature_extractor.feature_names,
            'feature_count': len(self.feature_extractor.feature_names),
            'dataset_stats': {
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df),
                'total_size': len(full_df)
            },
            'label_distribution': {
                'train': train_df['label'].value_counts().to_dict(),
                'val': val_df['label'].value_counts().to_dict(),
                'test': test_df['label'].value_counts().to_dict()
            },
            'processing_params': {
                'sample_every_n_plies': self.pgn_processor.sample_every_n_plies,
                'min_elo': self.pgn_processor.min_elo,
                'max_elo': self.pgn_processor.max_elo,
                'balance_results': self.balance_results,
                'max_positions_per_result': self.max_positions_per_result
            }
        }
        
        with open(self.processed_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature extractor for later use
        joblib.dump(self.feature_extractor, self.processed_dir / "feature_extractor.joblib")
        
        logger.info(f"Datasets saved to {self.processed_dir}")
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    def build_dataset(self, pgn_files: List[str], max_games_per_file: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Complete pipeline to build dataset from PGN files."""
        logger.info("Starting dataset building pipeline...")
        
        # Step 1: Extract positions from PGN files
        positions = self.process_pgn_files(pgn_files, max_games_per_file)
        
        if not positions:
            raise ValueError("No positions extracted from PGN files")
        
        # Step 2: Convert to DataFrame with features
        df = self.positions_to_dataframe(positions)
        
        # Step 3: Balance dataset if requested
        df = self.balance_dataset(df)
        
        # Step 4: Split dataset
        train_df, val_df, test_df = self.split_dataset(df)
        
        # Step 5: Save everything
        self.save_dataset(train_df, val_df, test_df)
        
        logger.info("Dataset building complete!")
        return train_df, val_df, test_df
    
    def load_existing_dataset(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load previously created dataset."""
        try:
            train_df = pd.read_parquet(self.processed_dir / "train.parquet")
            val_df = pd.read_parquet(self.processed_dir / "val.parquet")
            test_df = pd.read_parquet(self.processed_dir / "test.parquet")
            
            logger.info(f"Loaded existing dataset - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
            return train_df, val_df, test_df
        
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Dataset files not found. Please run build_dataset first. {e}")

def main():
    """CLI for dataset building."""
    parser = argparse.ArgumentParser(description="Build chess training dataset from PGN files")
    
    parser.add_argument("--pgn-files", nargs="+", required=True,
                       help="PGN files to process (should be in data/raw/)")
    parser.add_argument("--max-games-per-file", type=int, default=None,
                       help="Maximum games to process per PGN file")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory")
    parser.add_argument("--sample-every", type=int, default=2,
                       help="Sample positions every N plies")
    parser.add_argument("--min-elo", type=int, default=1000,
                       help="Minimum ELO rating")
    parser.add_argument("--max-elo", type=int, default=2200,
                       help="Maximum ELO rating")
    parser.add_argument("--no-balance", action="store_true",
                       help="Don't balance result classes")
    parser.add_argument("--max-per-result", type=int, default=50000,
                       help="Maximum positions per result class")
    
    args = parser.parse_args()
    
    # Build dataset
    builder = ChessDatasetBuilder(
        data_dir=args.data_dir,
        sample_every_n_plies=args.sample_every,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        balance_results=not args.no_balance,
        max_positions_per_result=args.max_per_result
    )
    
    train_df, val_df, test_df = builder.build_dataset(args.pgn_files, args.max_games_per_file)
    
    # Print summary statistics
    print("\n=== Dataset Summary ===")
    print(f"Training positions: {len(train_df):,}")
    print(f"Validation positions: {len(val_df):,}")
    print(f"Test positions: {len(test_df):,}")
    print(f"Total positions: {len(train_df) + len(val_df) + len(test_df):,}")
    
    print(f"\nFeature count: {len(builder.feature_extractor.feature_names)}")
    
    print(f"\nLabel distribution (train):")
    label_counts = train_df['label'].round(1).value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "Loss" if label == 0.0 else "Draw" if label == 0.5 else "Win"
        print(f"  {label_name} ({label}): {count:,}")

if __name__ == "__main__":
    main()