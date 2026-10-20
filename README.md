## Demo: trial-wise correlation embedding

Convert neural activity with shape **cells × time × trials** into a feature vector for each trial.

The demo uses the repository's existing functions to:

1. Compute a cell-by-cell correlation matrix across time for each trial.
2. Compute pairwise Riemannian distances between trial correlation matrices.
3. Construct an adaptive diffusion kernel and obtain diffusion coordinates.

### Dependencies

```bash
pip install numpy scipy scikit-learn
```

### Run with your data

Save your activity array as a `.npy` file with shape `(n_cells, n_time, n_trials)`:

```python
import numpy as np

np.save("activity.npy", data)
```

From the repository root, run:

```bash
python examples/demo_trial_embedding.py activity.npy --output embedding.npy
```

The saved array has shape `(n_trials, 10)`: one row per trial and one column per diffusion coordinate.

To request a different number of features:

```bash
python examples/demo_trial_embedding.py activity.npy --n-components 5 --output embedding.npy
```

### Use from Python

```python
from examples.demo_trial_embedding import embed_trials

features, distances = embed_trials(data, n_components=10)

# features.shape:  (n_trials, 10)
# distances.shape: (n_trials, n_trials)
```

### Try synthetic data

Run without an input file:

```bash
python examples/demo_trial_embedding.py
```

This generates synthetic activity and prints the input and output shapes.

### Data requirements

- Use the same cells in the same order across all trials.
- Input values must be finite, with no NaN or Inf.
- Cell traces must have nonzero variance within every trial.
- The number of trials must exceed the requested number of features.

### Implementation notes

Correlations are computed using `get_corr_matrix`. Riemannian distances and embedding are computed using `get_diffusion_embedding` in `riemannian` mode.

The distance routine uses a common rank selected from the trial matrices using an eigenvalue threshold of `0.01`. The diffusion kernel uses a separate bandwidth for each trial, estimated from its nearest-neighbor distances.

The embedding routine removes the leading eigenvector and weights the remaining eigenvectors by their eigenvalues. The demo selects the requested number of coordinates.

The default kernel uses a sparse nearest-neighbor graph. Check graph connectivity when interpreting the embedding, since disconnected components can produce additional trivial eigenvectors.
