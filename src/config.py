from pathlib import Path
 
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
RESULT_DIR = ROOT / "results"
 
TOP_LEVEL = ["PhysicalObject", "State", "Process", "Activity", "Property"]
SPLIT_SEED = 13                 
SEEDS = [0, 1, 2, 3, 4]         
BUDGET_GRID = [0.05, 0.10, 0.20, 0.40, 0.60]   # 占 train pool 总 token 的比例
THRESHOLDS = [0.90, 0.95, 0.98]

TRAIN = dict(max_len=48, batch_size=16, lr=5e-5, min_steps=300, min_epochs=5)
