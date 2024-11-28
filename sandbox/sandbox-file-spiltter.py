from tqdm import tqdm
from data_processing.augmenters import file_splitter

file_path = "./data/ROOTS.txt"
target_dir = "./data/segments"

file_splitter(
    data=file_path,
    target_dir=target_dir,
    split_threshold = 200_000, # 200k lines per file segment
    write_frequency = 2_000,
    file_encoding="utf-8",
    verbose=True
)     