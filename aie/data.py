"""Toy datasets reproducing the figures in Stone (2020).

These are kept tiny so notebooks run instantly without external data
downloads. ``mnist()`` optionally loads the real MNIST digits (downloaded
once and cached under ``data/``).
"""

from __future__ import annotations

import numpy as np


def two_class_2d(n_per_class: int = 100, seed: int = 0):
    """Two linearly separable classes split by x_2 = 0 (Fig 3.2).

    Returns
    -------
    X : (2*n_per_class, 2) array of input vectors
    t : (2*n_per_class,)  array of target labels in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    # Class C_1: x_2 ≤ 0  → target -1
    x1 = rng.uniform(-2, 2, n_per_class)
    y1 = rng.uniform(-2, 0, n_per_class)
    # Class C_2: x_2 > 0  → target +1
    x2 = rng.uniform(-2, 2, n_per_class)
    y2 = rng.uniform(0, 2, n_per_class)
    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    t = np.concatenate([-np.ones(n_per_class), np.ones(n_per_class)])
    perm = rng.permutation(len(t))
    return X[perm], t[perm]


def xor_dataset() -> tuple[np.ndarray, np.ndarray]:
    """The four XOR associations (Eq 3.5–3.6)."""
    X = np.array(
        [
            [-1.0, -1.0],
            [+1.0, +1.0],
            [-1.0, +1.0],
            [+1.0, -1.0],
        ]
    )
    t = np.array([-1.0, -1.0, +1.0, +1.0])
    return X, t


def xor_dataset_01() -> tuple[np.ndarray, np.ndarray]:
    """XOR with 0/1 inputs and targets (the form used by the Fig 4.5 weights)."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    t = np.array([0.0, 1.0, 1.0, 0.0])
    return X, t


def synthetic_face(size: int = 30) -> np.ndarray:
    """A simple synthetic 'photograph' for the Ch 2 autoassociator demo.

    Two black eyes and a smile on a grey background. Pixel values in [0, 1].
    """
    img = np.full((size, size), 0.5)
    # Eyes
    eye_y = size // 3
    img[eye_y : eye_y + 2, size // 4 : size // 4 + 3] = 0.0
    img[eye_y : eye_y + 2, 3 * size // 4 - 3 : 3 * size // 4] = 0.0
    # Smile (a thin curve)
    smile_y = 2 * size // 3
    for dx in range(-size // 4, size // 4 + 1):
        x = size // 2 + dx
        y = smile_y + int(0.05 * dx * dx)
        if 0 <= y < size:
            img[y, x] = 0.0
    return img


def synthetic_diamond(size: int = 30) -> np.ndarray:
    """A second pattern (rotated square) used alongside `synthetic_face`."""
    img = np.full((size, size), 0.5)
    c = size // 2
    radius = size // 3
    for i in range(size):
        for j in range(size):
            if abs(i - c) + abs(j - c) <= radius:
                img[i, j] = 1.0
    return img


def add_noise(img: np.ndarray, noise_level: float = 0.4, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise to an image and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noisy = img + rng.normal(0.0, noise_level, img.shape)
    return np.clip(noisy, 0.0, 1.0)


# 5×7 bitmap glyphs for the digits 0–9 (one string per row, '#' = ink).
_DIGIT_GLYPHS = {
    0: [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    1: ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    2: [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    3: ["#####", "...#.", "..#..", "...#.", "....#", "#...#", ".###."],
    4: ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    5: ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    6: ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    7: ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    8: [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    9: [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
}


def digit_glyphs() -> np.ndarray:
    """The ten clean 7×5 digit templates as a (10, 7, 5) array in {0, 1}."""
    return np.array(
        [[[c == "#" for c in row] for row in _DIGIT_GLYPHS[d]] for d in range(10)],
        dtype=float,
    )


def digits(
    n_per_class: int = 100,
    size: int = 12,
    noise: float = 0.1,
    max_shift: int = 2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """A small offline stand-in for MNIST.

    Each image is a 7×5 digit glyph placed on a ``size``×``size`` canvas,
    centred then randomly shifted by up to ``max_shift`` pixels in each
    direction, with Gaussian pixel noise added and clipped to [0, 1].

    Returns
    -------
    X : (10 · n_per_class, size, size) float array in [0, 1]
    y : (10 · n_per_class,) int array of labels 0–9, shuffled
    """
    if size < 7 + 2 * max_shift:
        raise ValueError("size too small for glyph plus shifts")
    rng = np.random.default_rng(seed)
    glyphs = digit_glyphs()
    n = 10 * n_per_class
    X = np.zeros((n, size, size))
    y = np.repeat(np.arange(10), n_per_class)
    r0, c0 = (size - 7) // 2, (size - 5) // 2
    for i in range(n):
        dr, dc = rng.integers(-max_shift, max_shift + 1, size=2)
        X[i, r0 + dr : r0 + dr + 7, c0 + dc : c0 + dc + 5] = glyphs[y[i]]
    X = np.clip(X + rng.normal(0.0, noise, X.shape), 0.0, 1.0)
    perm = rng.permutation(n)
    return X[perm], y[perm]


# ---------------------------------------------------------------------------
# Real MNIST (optional; requires a one-off ~11 MB download)
# ---------------------------------------------------------------------------

MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
_MNIST_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def read_idx(path) -> np.ndarray:
    """Read a (optionally gzipped) IDX file, the format MNIST is distributed in.

    The header is two zero bytes, a type code (0x08 = uint8), the number of
    dimensions, then one big-endian uint32 per dimension.
    """
    import gzip
    from pathlib import Path

    raw = Path(path).read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip magic number
        raw = gzip.decompress(raw)
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0 or raw[2] != 0x08:
        raise ValueError(f"{path} is not an unsigned-byte IDX file")
    ndim = raw[3]
    shape = tuple(int.from_bytes(raw[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path}: header shape {shape} does not match {data.size} bytes")
    return data.reshape(shape)


def _default_data_dir():
    import os
    from pathlib import Path

    env = os.environ.get("AIE_DATA_DIR")
    if env:
        return Path(env)
    repo = Path(__file__).resolve().parents[1]
    if (repo / "pyproject.toml").exists():  # running from a source checkout
        return repo / "data"
    return Path.home() / ".cache" / "aie"   # pip-installed (e.g. on Colab)


def mnist(
    split: str = "train",
    data_dir=None,
    download: bool = True,
    n: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the MNIST handwritten digits (LeCun et al. 1998; book Ch 9).

    Files are cached in ``<data_dir>/mnist/`` and downloaded on first use. The
    default ``data_dir`` is ``$AIE_DATA_DIR`` if set, else the repo's ``data/``
    folder in a source checkout, else ``~/.cache/aie``.

    Parameters
    ----------
    split     : "train" (60,000 images) or "test" (10,000 images)
    data_dir  : cache directory (see above for the default)
    download  : fetch missing files from a public mirror
    n         : if given, return only the first ``n`` examples
    normalize : scale pixels to [0, 1] float64 (else raw uint8)

    Returns
    -------
    X : (N, 28, 28) images
    y : (N,) integer labels 0–9
    """
    from pathlib import Path

    if split not in _MNIST_FILES:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = (Path(data_dir) if data_dir is not None else _default_data_dir()) / "mnist"
    paths = [root / name for name in _MNIST_FILES[split]]
    for p in paths:
        if not p.exists():
            if not download:
                raise FileNotFoundError(f"{p} not found (pass download=True to fetch it)")
            _download(p)
    X, y = read_idx(paths[0]), read_idx(paths[1])
    if len(X) != len(y):
        raise ValueError("MNIST image and label files have different lengths")
    if n is not None:
        X, y = X[:n], y[:n]
    X = X.astype(np.float64) / 255.0 if normalize else X
    return X, y.astype(np.int64)


def _download(path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for base in MNIST_MIRRORS:
        tmp = path.with_suffix(path.suffix + ".part")
        try:
            urllib.request.urlretrieve(base + path.name, tmp)
            read_idx(tmp)  # validate before committing to the cache
            tmp.replace(path)
            return
        except Exception as e:  # noqa: BLE001 - try the next mirror
            errors.append(f"{base}: {e}")
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not download {path.name}:\n" + "\n".join(errors))
