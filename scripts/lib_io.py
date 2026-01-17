from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


def iter_parquet_batches(
    parquet_path: str | Path,
    columns: list[str] | None = None,
    batch_size: int = 256_000,
) -> Iterable[dict[str, np.ndarray]]:
    """Iterate parquet file in record batches as numpy arrays."""

    pf = pq.ParquetFile(str(parquet_path))
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        out: dict[str, np.ndarray] = {}
        for name in batch.schema.names:
            out[name] = batch.column(name).to_numpy(zero_copy_only=False)
        yield out


def unique_seq_ix(parquet_path: str | Path) -> np.ndarray:
    """Extract unique seq_ix values without loading entire parquet."""

    seen: set[int] = set()
    for batch in iter_parquet_batches(parquet_path, columns=["seq_ix"]):
        for v in np.unique(batch["seq_ix"]):
            seen.add(int(v))
    return np.asarray(sorted(seen), dtype=np.int64)
