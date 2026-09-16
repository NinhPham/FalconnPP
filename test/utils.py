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

from pathlib import Path
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

    norms = np.linalg.norm(X, axis=0)
    print(np.allclose(norms, 1.0))

    norms = np.linalg.norm(X, axis=1)

    print("min norm:", norms.min())
    print("max norm:", norms.max())
    print("max error:", np.max(np.abs(norms - 1.0)))

    bad = np.where(~np.isclose(norms, 1.0))[0]

    print("number of bad vectors:", len(bad))
    print("first bad indices:", bad[:20])
    print("their norms:", norms[bad[:20]])

def normalize_bin(input_path, output_path, num_rows, num_cols, dtype=np.float32, batch_size=100000):
    # Read original binary file
    X = np.memmap(input_path, dtype=dtype, mode="r", shape=(num_rows, num_cols))

    # Create output binary file on disk
    X_norm = np.memmap(output_path, dtype=dtype, mode="w+", shape=(num_rows, num_cols))

    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)
        block = X[start:end]
        norms = np.linalg.norm(block, axis=1, keepdims=True)

        # Avoid division by zero
        norms[norms == 0] = 1.0

        # Write normalized block directly to X_norm.bin
        X_norm[start:end] = block / norms

    # Make sure everything is written to disk
    X_norm.flush()

    #return X_norm

if __name__ == '__main__':


    # --------------------------------------------------------------------------------
    # Loading data set
    path = Path("~/Work/Datasets/ANNS/").expanduser()

    dataset_file = path / "Glove_X_1183514_200.bin"
    query_file = path / "Glove_Q_1000_200.bin"

    nx = 1183514
    nq = 1000
    d = 200

    normalize_bin(dataset_file, path / "Glove_X_norm_1183514_200.bin", num_rows=nx, num_cols=d, dtype = np.float32)
    normalize_bin(query_file, path / "Glove_Q_norm_1000_200.bin", num_rows=nq, num_cols=d, dtype = np.float32)