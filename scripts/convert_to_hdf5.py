#!/usr/bin/env python3

import argparse
import os
from typing import Tuple

import h5py
import numpy as np


def read_fbin(path: str) -> Tuple[np.ndarray, int, int]:
    """
    Read a DiskANN-style .fbin file:
      int32 num_vectors, int32 dim, followed by num_vectors*dim float32 values.
    Returns (matrix, n, d) where matrix is shape (n, d), dtype float32.
    """
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=2)
        if header.shape[0] != 2:
            raise ValueError(f"'{path}' is too small to be a valid .fbin (missing header).")
        n = int(header[0])
        d = int(header[1])
        if n <= 0 or d <= 0:
            raise ValueError(f"Invalid .fbin header in '{path}': n={n}, d={d}")

        expected = n * d
        data = np.fromfile(f, dtype=np.float32, count=expected)
        if data.size != expected:
            raise ValueError(
                f"'{path}' is truncated: expected {expected} float32s, got {data.size}."
            )

    return data.reshape(n, d), n, d


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert .fbin base/query files into an HDF5 file with datasets "
            "'train' and 'test' (the format expected by experimental/panther/k-means)."
        )
    )
    ap.add_argument("--base", required=True, help="Path to base vectors (.fbin).")
    ap.add_argument("--query", required=True, help="Path to query vectors (.fbin).")
    ap.add_argument("--out", required=True, help="Output .hdf5 path.")
    ap.add_argument(
        "--train-key", default="train", help="HDF5 dataset name for base vectors."
    )
    ap.add_argument(
        "--test-key", default="test", help="HDF5 dataset name for query vectors."
    )
    ap.add_argument(
        "--compression",
        default="gzip",
        help="HDF5 compression (e.g. 'gzip', 'lzf', or 'none').",
    )
    ap.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="Compression level for gzip (ignored otherwise).",
    )
    args = ap.parse_args()

    base, nb, db = read_fbin(args.base)
    query, nq, dq = read_fbin(args.query)
    if db != dq:
        raise ValueError(f"Dim mismatch: base d={db}, query d={dq}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    compression = None if args.compression.lower() == "none" else args.compression
    compression_opts = args.compression_level if compression == "gzip" else None

    with h5py.File(args.out, "w") as f:
        f.create_dataset(
            args.train_key,
            data=base,
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
        )
        f.create_dataset(
            args.test_key,
            data=query,
            dtype=np.float32,
            compression=compression,
            compression_opts=compression_opts,
        )
        f.attrs["source_format"] = "fbin"
        f.attrs["base_path"] = os.path.basename(args.base)
        f.attrs["query_path"] = os.path.basename(args.query)
        f.attrs["n_train"] = nb
        f.attrs["n_test"] = nq
        f.attrs["dim"] = db

    print(f"Wrote HDF5: {args.out}")
    print(f"  {args.train_key}: shape={base.shape} dtype=float32")
    print(f"  {args.test_key}:  shape={query.shape} dtype=float32")


if __name__ == "__main__":
    main()
