"""Test-set image loader.

Resolves the list of (quadrat_id, image_path) pairs from either:
  1. A test metadata CSV (preferred - maps quadrat_id to filename explicitly).
  2. A plain directory scan (fallback - quadrat_id inferred from filename stem).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .config import PathConfig

logger = logging.getLogger(__name__)

# Image extensions to consider when scanning a directory.
_IMAGE_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_test_items(paths: PathConfig) -> list[tuple[str, Path]]:
    """Return a list of (quadrat_id, image_path) pairs for the test set.

    Loading strategy (in order):
      1. If ``paths.test_csv`` is provided and exists, parse it.
      2. Otherwise scan ``paths.image_dir`` recursively for image files.

    Parameters
    ----------
    paths : PathConfig
        Inference path configuration.

    Returns
    -------
    list[tuple[str, Path]]
        List of ``(quadrat_id, absolute_image_path)`` tuples, sorted by
        quadrat_id for reproducibility.

    Raises
    ------
    FileNotFoundError
        If ``paths.image_dir`` does not exist.
    ValueError
        If no images could be found.
    """
    # plantclef: RAM-Disk Satiation for Inference
    # We prefer the high-speed /dev/shm path if the images are cached there
    image_dir = paths.image_dir
    ram_disk_path = Path("/dev/shm/images")
    if ram_disk_path.exists():
        try:
            if any(ram_disk_path.iterdir()):
                print(f"[plantclef] Inference: Using RAM-Disk for zero-latency I/O.")
                image_dir = ram_disk_path
        except (PermissionError, StopIteration):
            pass

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if paths.test_csv is not None and paths.test_csv.exists():
        items = _load_from_csv(paths.test_csv, image_dir)
    else:
        if paths.test_csv is not None:
            logger.warning(
                "Test CSV not found at %s - falling back to directory scan.",
                paths.test_csv,
            )
        items = _load_from_directory(image_dir)

    if not items:
        raise ValueError(
            f"No test images found. image_dir={paths.image_dir}, "
            f"test_csv={paths.test_csv}"
        )

    items = sorted(items, key=lambda t: t[0])
    logger.info("Loaded %d test images.", len(items))
    return items


def _load_from_csv(
    csv_path: Path, image_dir: Path
) -> list[tuple[str, Path]]:
    """Parse the test metadata CSV.

    The CSV must have a ``quadrat_id`` column.  The image filename is resolved
    via (in order of preference):
      - A ``filename`` column.
      - An ``image_name`` column.
      - Using ``quadrat_id`` as the stem and searching ``image_dir``.

    Parameters
    ----------
    csv_path : Path
        Path to the test metadata CSV.
    image_dir : Path
        Root directory to search for image files.

    Returns
    -------
    list[tuple[str, Path]]
        List of ``(quadrat_id, image_path)`` pairs for images that exist on disk.

    Raises
    ------
    ValueError
        If 'quadrat_id' column is missing from CSV.
    """
    # Try semicolon first (PlantCLEF standard), then fallback
    try:
        df = pd.read_csv(csv_path, sep=';')
        if "quadrat_id" not in df.columns and "image_name" not in df.columns:
             # If mapping fails, try auto-detect
             df = pd.read_csv(csv_path, sep=None, engine='python')
    except Exception:
        df = pd.read_csv(csv_path, sep=None, engine='python')

    if "quadrat_id" not in df.columns:
        # Try common alternatives
        for alt in ("plot_id", "image_id", "id"):
            if alt in df.columns:
                df = df.rename(columns={alt: "quadrat_id"})
                logger.warning(
                    "Column 'quadrat_id' not found in %s; using '%s' instead.",
                    csv_path,
                    alt,
                )
                break
        else:
            raise ValueError(
                f"Cannot find a 'quadrat_id' column in {csv_path}. "
                f"Available columns: {list(df.columns)}"
            )

    items: list[tuple[str, Path]] = []
    missing = 0

    for _, row in df.iterrows():
        qid = str(row["quadrat_id"])
        image_path = _resolve_image_path(qid, row, image_dir)
        if image_path is None:
            missing += 1
            logger.debug("Image not found for quadrat_id=%s; skipping.", qid)
        else:
            items.append((qid, image_path))

    if missing:
        logger.warning(
            "%d / %d images from the CSV could not be found on disk.",
            missing,
            len(df),
        )

    return items


def _resolve_image_path(
    quadrat_id: str,
    row: pd.Series,
    image_dir: Path,
) -> Path | None:
    """Try to locate an image file for a given CSV row.

    Checks explicit filename columns first, then falls back to searching
    ``image_dir`` for any file whose stem matches ``quadrat_id``.

    Parameters
    ----------
    quadrat_id : str
        The unique identifier for the quadrat.
    row : pd.Series
        The CSV row containing metadata for the quadrat.
    image_dir : Path
        Root directory to search for image files.

    Returns
    -------
    Path | None
        The path to the image file if found, otherwise None.
    """
    # 1. Explicit filename column
    for col in ("filename", "image_name", "file_name", "image_path"):
        if col in row.index and pd.notna(row[col]):
            candidate = image_dir / str(row[col])
            logger.debug("[PathCheck] Column: %s, Candidate: %s", col, candidate)
            if candidate.exists():
                return candidate
            # Maybe it's an absolute path already
            candidate2 = Path(str(row[col]))
            if candidate2.exists():
                return candidate2

    # 2. Stem-based search in image_dir
    for ext in _IMAGE_EXTENSIONS:
        candidate = image_dir / f"{quadrat_id}{ext}"
        if candidate.exists():
            return candidate

    # 3. Recursive search (slower - only used as last resort)
    for path in image_dir.rglob(f"{quadrat_id}.*"):
        if path.suffix.lower() in _IMAGE_EXTENSIONS:
            return path

    return None


def _load_from_directory(image_dir: Path) -> list[tuple[str, Path]]:
    """Scan ``image_dir`` recursively and treat each image's stem as its quadrat_id.

    Parameters
    ----------
    image_dir : Path
        Root directory to scan.

    Returns
    -------
    list[tuple[str, Path]]
        List of ``(quadrat_id, image_path)`` pairs.
    """
    items: list[tuple[str, Path]] = []
    for path in sorted(image_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            items.append((path.stem, path.resolve()))
    logger.info(
        "Directory scan of %s found %d images.", image_dir, len(items)
    )
    return items


def iter_images(
    items: list[tuple[str, Path]]
) -> Iterator[tuple[str, Path]]:
    """Iterate over test items, logging progress every 50 images.

    Parameters
    ----------
    items : list[tuple[str, Path]]
        Output of :func:`load_test_items`.

    Yields
    ------
    tuple[str, Path]
        ``(quadrat_id, image_path)`` tuples.
    """
    n = len(items)
    for i, (qid, path) in enumerate(items):
        if i % 50 == 0:
            logger.info("Processing image %d / %d: %s", i + 1, n, qid)
        yield qid, path
