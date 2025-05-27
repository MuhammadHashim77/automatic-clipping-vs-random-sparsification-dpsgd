from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

DATA_DIR    = Path("data") 

df = pd.read_csv(DATA_DIR / "train-rle.csv")

# print column names
print("Column names in the DataFrame:")
for column in df.columns:
    print(column)

