import os

# must set before import

os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["FAISS_NUM_THREADS"] = "8"

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["FAISS_NUM_THREADS"] = "1"

# import sDbscan
import faiss
import numpy as np
import math

def mmap_bin(bin_path, num_rows, num_cols, dtype=np.float32):
    return np.memmap(bin_path, dtype=dtype, mode='r', shape=(num_rows, num_cols)) # read-only mode
    # return np.memmap(bin_path, dtype=dtype, mode='c', shape=(num_rows, num_cols)) # copy-on-write mode

def inspect_data(X):
    print(f"X.shape      = {X.shape}")
    print(f"X.dtype      = {X.dtype}")
    print(f"X size       = {X.nbytes / 1024**3:.3f} GB")
    print(f"NaNs         = {np.isnan(X).sum()}")
    print(f"Infs         = {np.isinf(X).sum()}")

    _, counts = np.unique(X, axis=0, return_counts=True)

    print(f"Unique rows  = {len(counts)}")
    print(f"Duplicates   = {len(X) - len(counts)}")
    print(f"Duplicated vectors = {(counts > 1).sum()}")


