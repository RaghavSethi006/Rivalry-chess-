"""
Script to download and extract Lichess database files.
"""
import requests
import zstandard as zstd
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import io
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LichessDownloader:
    """Downloads and extracts Lichess database files."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filename: str) -> Path:
        """Download a file with progress bar."""
        filepath = self.raw_dir / filename
        
        if filepath.exists():
            logger.info(f"File {filename} already exists. Skipping download.")
            return filepath
        
        logger.info(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Downloaded {filename} ({total_size / (1024*1024):.1f} MB)")
        return filepath
    
    def extract_zst_to_pgn(self, zst_path: Path, max_games: Optional[int] = None) -> Path:
        """Extract .zst file to .pgn file, optionally limiting number of games."""
        pgn_path = zst_path.with_suffix('.pgn')
        
        if pgn_path.exists():
            logger.info(f"PGN file {pgn_path.name} already exists. Skipping extraction.")
            return pgn_path
        
        logger.info(f"Extracting {zst_path.name} to {pgn_path.name}")
        
        # Create decompressor
        dctx = zstd.ZstdDecompressor()
        
        games_extracted = 0
        
        with open(zst_path, 'rb') as compressed_file:
            with dctx.stream_reader(compressed_file) as reader:
                with open(pgn_path, 'w', encoding='utf-8') as output_file:
                    text_reader = io.TextIOWrapper(reader, encoding='utf-8')
                    
                    current_game = []
                    
                    for line in text_reader:
                        current_game.append(line)
                        
                        # Check if this is the end of a game
                        if line.strip() and not line.startswith('[') and not line[0].isdigit():
                            # This is likely the result line (1-0, 0-1, 1/2-1/2)
                            if any(result in line for result in ['1-0', '0-1', '1/2-1/2', '*']):
                                # Write the complete game
                                output_file.writelines(current_game)
                                output_file.write('\n')  # Add separator
                                
                                games_extracted += 1
                                current_game = []
                                
                                if games_extracted % 10000 == 0:
                                    logger.info(f"Extracted {games_extracted} games...")
                                
                                if max_games and games_extracted >= max_games:
                                    logger.info(f"Reached max games limit: {max_games}")
                                    break
        
        logger.info(f"Extraction complete. {games_extracted} games extracted to {pgn_path.name}")
        return pgn_path

    def download_and_extract_lichess(self, year_month: str = "2024-01", max_games: Optional[int] = None) -> Path:
        """Download and extract a Lichess database file."""
        
        # Construct URL and filename
        filename = f"lichess_db_standard_rated_{year_month}.pgn.zst"
        url = f"https://database.lichess.org/standard/{filename}"
        
        try:
            # Download the compressed file
            zst_path = self.download_file(url, filename)
            
            # Extract to PGN
            pgn_path = self.extract_zst_to_pgn(zst_path, max_games)
            
            return pgn_path
            
        except Exception as e:
            logger.error(f"Failed to download/extract {filename}: {e}")
            raise

def main():
    """CLI for downloading Lichess data."""
    parser = argparse.ArgumentParser(description="Download and extract Lichess database files")
    
    parser.add_argument("--year-month", default="2024-01",
                       help="Year-month to download (e.g., 2024-01)")
    parser.add_argument("--max-games", type=int, default=50000,
                       help="Maximum games to extract (for testing)")
    parser.add_argument("--data-dir", default="data",
                       help="Data directory")
    
    args = parser.parse_args()
    
    downloader = LichessDownloader(args.data_dir)
    
    try:
        pgn_path = downloader.download_and_extract_lichess(
            year_month=args.year_month,
            max_games=args.max_games
        )
        
        print(f"\n✅ Success!")
        print(f"PGN file ready: {pgn_path}")
        print(f"\nNext step: Run the dataset builder:")
        print(f"python ml/make_dataset.py --pgn-files {pgn_path.name} --max-games-per-file 10000")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nAlternative: If download fails, you can:")
        print("1. Manually download from https://database.lichess.org/")
        print("2. Place the .pgn.zst file in data/raw/")
        print("3. Run this script to extract it")

if __name__ == "__main__":
    main()