"""Offline tests for the MNIST loader using tiny synthetic IDX files."""

import gzip
import struct

import numpy as np
import pytest

from aie.data import mnist, read_idx


def _write_idx(path, arr: np.ndarray, compress: bool = True) -> None:
    header = bytes([0, 0, 0x08, arr.ndim]) + b"".join(struct.pack(">I", d) for d in arr.shape)
    payload = header + arr.astype(np.uint8).tobytes()
    path.write_bytes(gzip.compress(payload) if compress else payload)


@pytest.mark.parametrize("compress", [True, False])
def test_read_idx_roundtrip(tmp_path, compress):
    arr = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    p = tmp_path / "x.idx"
    _write_idx(p, arr, compress)
    np.testing.assert_array_equal(read_idx(p), arr)


def test_read_idx_rejects_bad_files(tmp_path):
    p = tmp_path / "bad"
    p.write_bytes(b"\x00\x00\x0d\x01\x00\x00\x00\x02ab")  # float type code
    with pytest.raises(ValueError):
        read_idx(p)
    p.write_bytes(bytes([0, 0, 8, 1, 0, 0, 0, 5]) + b"abc")  # truncated
    with pytest.raises(ValueError):
        read_idx(p)


def test_mnist_loads_cached_files(tmp_path):
    rng = np.random.default_rng(0)
    imgs = rng.integers(0, 256, (6, 28, 28), dtype=np.uint8)
    labels = rng.integers(0, 10, 6, dtype=np.uint8)
    d = tmp_path / "mnist"
    d.mkdir()
    _write_idx(d / "t10k-images-idx3-ubyte.gz", imgs)
    _write_idx(d / "t10k-labels-idx1-ubyte.gz", labels)

    X, y = mnist("test", data_dir=tmp_path, download=False)
    assert X.shape == (6, 28, 28) and X.dtype == np.float64
    np.testing.assert_allclose(X, imgs / 255.0)
    np.testing.assert_array_equal(y, labels)

    Xr, yr = mnist("test", data_dir=tmp_path, download=False, n=4, normalize=False)
    assert Xr.dtype == np.uint8 and len(Xr) == len(yr) == 4


def test_mnist_missing_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        mnist("train", data_dir=tmp_path, download=False)
    with pytest.raises(ValueError):
        mnist("valid", data_dir=tmp_path, download=False)
