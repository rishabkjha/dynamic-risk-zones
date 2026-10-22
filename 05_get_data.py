"""Step 5: download the ETH/UCY pedestrian trajectory datasets (14 MB) into data/raw."""
from dynnav import data

n = data.download("data/raw")
print(f"{n} files in data/raw. Next: python 06_train_predictor.py")
