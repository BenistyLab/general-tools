"""Embed trial-wise cell correlations using the general-tools functions.

Place this file in examples/ at the repository root.
Dependencies: numpy, scipy, scikit-learn.

Run with data saved as a NumPy array (cells, time, trials):
    python examples/demo_trial_embedding.py activity.npy --output embedding.npy

Run without a file to try synthetic data:
    python examples/demo_trial_embedding.py

The saved embedding has shape (trials, n_components). This demo uses the
repository's correlation, Riemannian distance, kernel, and embedding routines
without reimplementing them. The same cells must occur in the same order in
every trial.
"""

import argparse
from pathlib import Path
import sys

import numpy as np

# Folder names contain hyphens, so add them to the Python import path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "riemmanian-geometry"))
sys.path.insert(0, str(REPO_ROOT / "dim-reduction"))

from Riemannian_utils import get_corr_matrix
from Diffusion_emb_utils import get_diffusion_embedding


def embed_trials(data, n_components=10):
    """Return (features, distances) from cells x time x trials activity.

    features: (trials, n_components), eigenvalue-weighted coordinates.
    distances: (trials, trials), pairwise Riemannian distances.

    The repository selects a common rank using eigenvalues > 0.01, then
    computes PSD distances. Its Riemannian kernel uses adaptive bandwidths
    and a sparse nearest-neighbor graph by default. It removes the leading
    eigenvector before returning the embedding. Disconnected graphs can
    retain additional trivial components; inspect connectivity for your data.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("Expected data with shape (cells, time, trials).")
    n_cells, n_time, n_trials = data.shape
    if n_cells < 2 or n_time < 2:
        raise ValueError("Need at least two cells and two time points.")
    if not isinstance(n_components, (int, np.integer)) or not 1 <= n_components < n_trials:
        raise ValueError("n_components must be an integer between 1 and n_trials - 1.")
    if not np.isfinite(data).all():
        raise ValueError("Handle NaN/Inf values before running the demo.")
    if np.any(np.std(data, axis=1) == 0):
        raise ValueError(
            "A cell is constant within a trial. Select a common set of "
            "valid cells across all trials before computing correlations."
        )

    # Each call correlates cell traces over the time axis of one trial.
    correlations = np.stack([
        get_corr_matrix(data[:, :, trial])
        for trial in range(n_trials)
    ])  # (trials, cells, cells)

    # This function computes distances AND embedding internally.
    # window_length is the number of time samples used per correlation.
    embedding, distances = get_diffusion_embedding(
        correlations,
        window_length=n_time,
        mode="riemannian",
    )

    # Outputs include a batch axis: (1, components, trials), (1, trials, trials).
    features = embedding[0, :n_components, :].T
    distances = distances[0]
    if not np.isfinite(features).all() or not np.isfinite(distances).all():
        raise ValueError("Repository routines returned nonfinite results; inspect the data.")
    return features, distances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, help="Input .npy: cells x time x trials.")
    parser.add_argument("--n-components", type=int, default=10)
    parser.add_argument("--output", type=Path, help="Save embedding as a .npy file.")
    args = parser.parse_args()
    if args.output is not None and args.output.suffix != ".npy":
        parser.error("--output must have the .npy extension.")

    if args.data is None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((8, 100, 40))
        print("Using synthetic data for a shape/running example.")
    else:
        data = np.load(args.data, allow_pickle=False)

    features, distances = embed_trials(data, args.n_components)
    print(f"Input (cells, time, trials): {data.shape}")
    print(f"Riemannian distances (trials, trials): {distances.shape}")
    print(f"Embedding (trials, features): {features.shape}")
    if args.output is not None:
        np.save(args.output, features)
        print(f"Saved embedding to {args.output}")


if __name__ == "__main__":
    main()
