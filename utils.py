"""
utils.py — Shared utilities for the spaceflight–disease network separation pipeline.

Used by:
  1_disease_module_curation.ipynb
  2_separation_pipeline.ipynb
  3_spaceflight_disease_visualization.ipynb
  run_screening.py

Functions
---------
load(path)                           Deserialise a pickle file (joblib fallback).
save(obj, path)                      Serialise any object to a pickle file.
disease_to_filename(name, chkdir)    Convert a disease name to a safe checkpoint path.
simulate_parallel_runtime(...)       LPT-schedule job times across workers.
fit_empirical_model(empirical_log)   Fit a runtime model from observed (size, time) pairs.
build_symbol_to_entrez(symbols)      Map gene symbols → Entrez IDs using a local NCBI TSV.
build_ensembl_to_entrez(ensembl_ids) Map Ensembl IDs → Entrez IDs using a local NCBI TSV.
load_gene_info(path)                 Load and cache the NCBI gene_info TSV.
"""

from __future__ import annotations

import os
import pickle
import logging
from typing import Any, Callable, Iterable

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the NCBI gene_info flat file.
# Download once with:
#   wget -O data/input/gene_info/Homo_sapiens.gene_info.gz \
#        https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
#   gunzip data/input/gene_info/Homo_sapiens.gene_info.gz
# ---------------------------------------------------------------------------
GENE_INFO_PATH: str = "data/input/gene_info/Homo_sapiens.gene_info"

# Module-level cache so the TSV is parsed only once per Python session.
_gene_info_cache: "dict | None" = None


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save(obj: Any, path: str) -> None:
    """Serialise *obj* to *path* using pickle (protocol 4 for broad compat)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=4)


def load(path: str) -> Any:
    """Deserialise a pickle file.

    Falls back to *joblib* for legacy ``.pkl`` files written by older versions
    of the pipeline that used ``joblib.dump``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        try:
            import joblib  # type: ignore[import]
            return joblib.load(path)
        except Exception as exc:
            raise RuntimeError(f"Could not load {path} with pickle or joblib.") from exc


# ---------------------------------------------------------------------------
# Checkpoint filename helper
# ---------------------------------------------------------------------------

def disease_to_filename(name: str, checkpoint_dir: str = "data/output/checkpoints") -> str:
    """Return the full checkpoint filepath for *name*.

    Characters that are unsafe in filenames (``/ ( ) : ,``) and whitespace
    are replaced with underscores.

    Parameters
    ----------
    name:
        Disease name as stored in the ``disease_genes`` dictionary.
    checkpoint_dir:
        Directory that contains per-disease checkpoint files.

    Returns
    -------
    str
        Absolute-or-relative path ``<checkpoint_dir>/<safe_name>.pkl``.
    """
    safe = (
        name.replace("/", "_").replace(" ", "_")
            .replace("(", "").replace(")", "")
            .replace(":", "").replace(",", "")
    )
    return os.path.join(checkpoint_dir, f"{safe}.pkl")


# ---------------------------------------------------------------------------
# Runtime estimation helpers
# ---------------------------------------------------------------------------

def simulate_parallel_runtime(job_times: Iterable[float], n_workers: int) -> float:
    """Estimate wall-clock time using the *Longest Processing Time* (LPT) heuristic.

    Assigns each job (largest first) to the worker that finishes earliest,
    returning the makespan (time when the last worker finishes).

    Parameters
    ----------
    job_times:
        Predicted per-job runtimes in seconds.
    n_workers:
        Number of parallel workers.

    Returns
    -------
    float
        Estimated wall-clock seconds.
    """
    if n_workers <= 0:
        raise ValueError("n_workers must be a positive integer.")
    worker_finish = [0.0] * n_workers
    for t in sorted(job_times, reverse=True):
        earliest = min(range(n_workers), key=lambda w: worker_finish[w])
        worker_finish[earliest] += t
    return max(worker_finish)


def fit_empirical_model(
    empirical_log: list[tuple[int, float]],
) -> tuple[Callable[[int], float], str]:
    """Fit a runtime model from observed ``(gene_set_size, elapsed_seconds)`` pairs.

    Model selection:

    * **n ≥ 5** → quadratic least-squares (``a2·sz² + a1·sz + a0``).
    * **3 ≤ n < 5** → linear fit (``a1·sz + a0``).
    * **n < 3** → constant mean.

    The returned callable is clipped to a minimum of 1 second.

    Parameters
    ----------
    empirical_log:
        List of ``(size, elapsed_seconds)`` tuples collected during the run.

    Returns
    -------
    (predict_fn, label)
        *predict_fn(size)* → estimated seconds (≥ 1).
        *label* is a short human-readable description for logging.

    Notes
    -----
    FIX — redundant default-argument shadowing removed: the lambdas previously
    used ``_a2=a2``, ``_a1=a1``, ``_a0=a0`` (and analogues) to capture loop
    variables, which is only necessary inside a loop.  Because these lambdas
    are created once and immediately returned, the coefficients are already
    captured correctly by the enclosing scope closure.  The shadowing aliases
    have been removed to eliminate the confusing dual names.
    """
    n = len(empirical_log)
    if n == 0:
        return (lambda sz: 1.0), "no data"

    obs_sizes = np.array([s for s, _ in empirical_log], dtype=float)
    obs_times = np.array([t for _, t in empirical_log], dtype=float)

    if n >= 5:
        A = np.column_stack([obs_sizes ** 2, obs_sizes, np.ones(n)])
        coeffs, *_ = np.linalg.lstsq(A, obs_times, rcond=None)
        # BUG FIX: coeffs is a 1-D array; index directly instead of unpacking
        # via starred assignment, which silently drops residuals into *_.
        a2 = float(coeffs[0])
        a1 = float(coeffs[1])
        a0 = float(coeffs[2])
        return (
            lambda sz: max(a2 * sz ** 2 + a1 * sz + a0, 1.0),
            f"quad a2={a2:.2e} a1={a1:.4f}",
        )
    elif n >= 3:
        coeffs1d = np.polyfit(obs_sizes, obs_times, 1)
        a1 = float(coeffs1d[0])
        a0 = float(coeffs1d[1])
        return (
            lambda sz: max(a1 * sz + a0, 1.0),
            f"linear a1={a1:.4f} [n={n}]",
        )

    avg = float(obs_times.mean())
    # Constant model — gene set size is irrelevant with fewer than 3 observations.
    return (lambda sz: avg), f"mean={avg:.0f}s [n={n}]"


# ---------------------------------------------------------------------------
# Local NCBI gene_info ID mapping
# ---------------------------------------------------------------------------

def load_gene_info(path: str = GENE_INFO_PATH) -> dict:
    """Parse the NCBI *Homo_sapiens.gene_info* TSV and return a mapping dict.

    The file is cached in ``_gene_info_cache`` so it is only read once per
    Python session regardless of how many times this function is called.

    Returns
    -------
    dict with keys:
        ``"symbol_to_entrez"`` : dict[str, int]
            Official gene symbol → Entrez gene ID.
        ``"synonym_to_entrez"`` : dict[str, int]
            Each pipe-separated synonym → Entrez gene ID (official symbol wins
            on collision).
        ``"ensembl_to_entrez"`` : dict[str, int]
            Ensembl gene ID (no version suffix) → Entrez gene ID.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.  See module docstring for download command.
    """
    global _gene_info_cache
    if _gene_info_cache is not None:
        return _gene_info_cache

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"NCBI gene_info file not found at '{path}'.\n"
            "Download it with:\n"
            "  wget -O data/input/gene_info/Homo_sapiens.gene_info.gz \\\n"
            "       https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/"
            "Homo_sapiens.gene_info.gz\n"
            "  gunzip data/input/gene_info/Homo_sapiens.gene_info.gz"
        )

    symbol_to_entrez:  dict[str, int] = {}
    synonym_to_entrez: dict[str, int] = {}
    ensembl_to_entrez: dict[str, int] = {}

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            # Columns (0-based): tax_id, GeneID, Symbol, LocusTag,
            #                     Synonyms, dbXrefs, ...
            try:
                entrez = int(parts[1])
            except ValueError:
                continue

            symbol = parts[2]
            if symbol and symbol != "-":
                symbol_to_entrez[symbol] = entrez

            synonyms_field = parts[4]
            if synonyms_field and synonyms_field != "-":
                for syn in synonyms_field.split("|"):
                    syn = syn.strip()
                    if syn and syn not in symbol_to_entrez:
                        synonym_to_entrez[syn] = entrez

            dbxrefs_field = parts[5]
            if dbxrefs_field and dbxrefs_field != "-":
                for xref in dbxrefs_field.split("|"):
                    if xref.startswith("Ensembl:"):
                        ensembl_id = xref[len("Ensembl:"):]
                        # Strip version suffix (e.g. ENSG00000000001.3 → ENSG00000000001)
                        ensembl_id = ensembl_id.split(".")[0]
                        if ensembl_id:
                            ensembl_to_entrez[ensembl_id] = entrez

    _gene_info_cache = {
        "symbol_to_entrez":  symbol_to_entrez,
        "synonym_to_entrez": synonym_to_entrez,
        "ensembl_to_entrez": ensembl_to_entrez,
    }
    logger.info(
        "gene_info loaded: %d symbols, %d synonyms, %d Ensembl IDs",
        len(symbol_to_entrez),
        len(synonym_to_entrez),
        len(ensembl_to_entrez),
    )
    return _gene_info_cache


def build_symbol_to_entrez(
    symbols: Iterable[str],
    gene_info_path: str = GENE_INFO_PATH,
) -> dict[str, int]:
    """Map an iterable of gene symbols to Entrez IDs using the local NCBI file.

    Official symbol is tried first; then synonyms are used as fallback.
    Symbols that cannot be mapped are silently skipped.

    Parameters
    ----------
    symbols:
        Gene symbols to look up (e.g. from GWAS ``MAPPED_GENE`` column or
        JAXA gene list).
    gene_info_path:
        Path to the decompressed NCBI *Homo_sapiens.gene_info* file.

    Returns
    -------
    dict[str, int]
        ``{symbol: entrez_id}`` for every symbol that could be resolved.
    """
    gi = load_gene_info(gene_info_path)
    s2e  = gi["symbol_to_entrez"]
    syn2e = gi["synonym_to_entrez"]
    result: dict[str, int] = {}
    for sym in symbols:
        sym = sym.strip()
        if not sym:
            continue
        if sym in s2e:
            result[sym] = s2e[sym]
        elif sym in syn2e:
            result[sym] = syn2e[sym]
    return result


def build_ensembl_to_entrez(
    ensembl_ids: Iterable[str],
    gene_info_path: str = GENE_INFO_PATH,
) -> dict[str, int]:
    """Map an iterable of Ensembl gene IDs to Entrez IDs using the local NCBI file.

    Version suffixes (e.g. ``.13``) are stripped automatically.

    Parameters
    ----------
    ensembl_ids:
        Ensembl gene IDs to look up (e.g. from Inspiration4 row-0 column).
    gene_info_path:
        Path to the decompressed NCBI *Homo_sapiens.gene_info* file.

    Returns
    -------
    dict[str, int]
        ``{ensembl_id: entrez_id}`` for every ID that could be resolved.
    """
    gi = load_gene_info(gene_info_path)
    e2e = gi["ensembl_to_entrez"]
    result: dict[str, int] = {}
    for eid in ensembl_ids:
        eid = eid.strip().split(".")[0]  # strip version suffix
        if eid and eid in e2e:
            result[eid] = e2e[eid]
    return result
