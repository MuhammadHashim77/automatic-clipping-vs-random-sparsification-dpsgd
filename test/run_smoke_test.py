import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.train import run_baseline

if __name__ == "__main__":
    run_baseline(batch_size=16, epochs=5)
