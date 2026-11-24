# Neural data decoding

Shared MATLAB trial representations, diffusion embeddings, classification, and regression for neural activity
analyses across projects. Project data loading, experimental labels, run grouping,
and analysis orchestration belong in the individual project folders.

## Setup

Requires MATLAB with Statistics and Machine Learning Toolbox. MATLAB R2023a is
installed locally, but runtime verification is pending because license checkout
failed (License Manager Error -5). Source hashes and forwarding paths were checked;
MATLAB's Code Analyzer found no syntax errors in the embedding migration or tests.

```matlab
addpath('C:\Users\hadasbe\Benisty Lab Dropbox\Hadas Benisty\code\neural data decoding');
setupNeuralDataDecoding;

[accuracy, auc] = neuraldecoding.classification.logisticRegressionCV(X, labels, 5);
[r2, rmse, correlation] = neuraldecoding.regression.linearRegressionCV(X, target, 5);
```

Run setup once per MATLAB session, or from each project's startup script. Add
the repository root and use setup; avoid `genpath` on MATLAB package folders.
The package namespace prevents collisions with project functions of the same name.

## Functions

Matrix inputs use **trials (observations) x features**. Tensor inputs use
**sensors (neurons) x time samples x trials**. Labels/targets have one entry per
trial. Functions accept numerical features; they do not require a project-specific
data structure. The separate embedding package converts trial tensors into
representations and diffusion coordinates.

| Package | Function | Purpose |
| --- | --- | --- |
| `neuraldecoding.classification` | `logisticRegressionCV` | Binary logistic CV; mean accuracy and AUC |
| | `sensorLogisticCV` | Average a tensor time window; unregularized binary logistic CV |
| | `pairwiseClassifierCV` | Binary ridge logistic CV, plus scores for an unseen third class |
| | `pairwiseClassifierCVBalanced` | Pairwise CV with equal total class weight |
| | `pairwiseClassifierTrainTest` | Train/test masks for transfer between separate trial groups |
| | `nestedRepeatedL1SVM` | Nested multiclass L1 SVM CV with repeated undersampling |
| `neuraldecoding.regression` | `linearRegressionCV` | Linear CV; mean R2, RMSE, and correlation |
| | `timeResolvedRegressionCV` | One linear regression per time sample |
| | `slidingWindowRegressionCV` | Linear regression of time-window averages |
| | `trialIndexRegressionCV` | CV for normalized trial progress in [0, 1] |
| | `trialIndexRegressionTrainTest` | Transfer regression for normalized trial progress |
| `neuraldecoding.embedding` | `centered_snapshot_all_representations` | Raw per-trial means, correlations, covariances, and centered SVD factors |
| | `computeSnapshotEmbeddings` | Diffusion embeddings for all eight representations, or a selected subset |
| | `computeMeanActivityEmbedding` | Mean-only embedding, without SVD or other representations |
| | `extractNeuronMatrices` | Independent covXt/corrXt matrices, per-trial ranks, and constant-neuron flags |
| | `computeNeuronMatrixEmbeddings` | Independent rank-aware covXt/corrXt distances and diffusion coordinates |
| | `pairwise_manifold_distances` | Euclidean, SPD/SPSD, and SO/frame distances between trials |
| | `dist2affinity` | Convert trial distances to Gaussian affinities |
| | `calcDiffusionMap` | Diffusion-map coordinates from an affinity matrix |

Use `help neuraldecoding.classification.sensorLogisticCV`, for example, for the
original function documentation and signatures.

## Preserved behavior

Classification and regression implementations were migrated byte for byte. The
embedding migration preserves the original calculations and default output fields,
qualifies shared dependencies, and adds representation selection and input checks.
Important
existing conventions:

- `pairwiseClassifierCVBalanced` calls positive-class probabilities
  `waterProbability` even when the positive label is something other than water.
- Trial-index regressors require targets in [0, 1] and select the first feature
  columns up to `maxComponents`; they are specialized, not general regressors.
- `timeResolvedRegressionCV` clips R2 at zero; sliding-window regression does not.
- Random-seed behavior follows each original function. Seed-controlled functions
  change MATLAB's global RNG; set `rng` before other CV functions for repeatability.
- `nestedRepeatedL1SVM` requires an explicit options struct containing
  `OuterFolds`, `InnerFolds`, `InnerRepeats`, `FinalRepeats`, `LambdaGrid`,
  `SelectionMetric`, and `RandomSeed`. Its commented argument block does not
  supply defaults. The implementation prints metrics and uses default ECOC coding;
  the documented `Coding` and `MakePlots` options are not implemented.
- CV splits individual trials. Callers must choose a method appropriate for
  temporally correlated trials or session-level evaluation. For transfer methods,
  supply disjoint training and test observations with aligned feature columns.
- Fit learned feature transformations using training data when evaluating held-out
  decoding. A decoder's fold-wise standardization does not make an embedding
  computed from all trials independent of the test data.

## Trial representations and embeddings

The embedding package is self-contained: it does **not** need `livne_slm` or
`roni` on the MATLAB path. Geometry, SVD orientation, and parameter helpers from
`roni` are included here; implementation helpers are private to the package.
The original files in `roni` are unchanged for its other users.

For `X` of size **neurons x time samples x trials**:

| Name | Representation for each trial | Distance |
| --- | --- | --- |
| `meanActivity` | Per-neuron mean over an inclusive sample window | Euclidean |
| `corrX` | Time x time correlation, treating neurons as observations | SPD/SPSD |
| `covX` | Time x time covariance, treating neurons as observations | SPD/SPSD |
| `corrXt` | Neurons x neurons correlation, treating time samples as observations | SPD/SPSD |
| `covXt` | Neurons x neurons covariance, treating time samples as observations | SPD/SPSD |
| `soU` | Left singular vectors of the temporally centered trial | SO / orthonormal-frame |
| `soV` | Right singular vectors in a common temporal contrast basis | SO / orthonormal-frame |
| `sv` | Diagonal matrix of singular values of the centered trial | Euclidean |

```matlab
% Existing behavior: all eight representations, then distances -> affinity
% -> diffusion map for each representation.
embedding = neuraldecoding.embedding.computeSnapshotEmbeddings(X, 20, [1 140]);

% Select which distances and embeddings to compute, in the requested order.
embedding = neuraldecoding.embedding.computeSnapshotEmbeddings( ...
    X, 20, [1 140], ["meanActivity", "corrXt", "soU"]);
features = embedding.psi.corrXt.'; % trials x diffusion coordinates

% Mean-only path avoids constructing SVD/correlation/covariance representations.
meanEmbedding = neuraldecoding.embedding.computeMeanActivityEmbedding(X, 20, [1 140]);
features = meanEmbedding.psi.';

% Raw trial representations, without distances or embedding:
[U, V, S, corrX, covX, corrXt, covXt, featureMeans, B, info, meanActivity] = ...
    neuraldecoding.embedding.centered_snapshot_all_representations(X, true, [1 140]);
```

Selecting only `covXt`, `corrXt`, and/or `meanActivity` in
`computeSnapshotEmbeddings` bypasses the shared centered-SVD requirement.
Selecting any of `soU`, `soV`, `sv`, `corrX`, or `covX` retains that requirement:
each centered trial must have rank `min(neurons, time-1)`. A failure of those
legacy representations can still stop a combined call; use the independent
neuron-matrix API below to calculate covXt/corrXt regardless of SVD failures.
The low-level raw-representation function also accepts `false` as its second
argument to allow rank-deficient trials.

```matlab
% No trial omission or centered-SVD requirement. Uses the supplied full tensor.
[matrices, info] = neuraldecoding.embedding.extractNeuronMatrices(X);
embedding = neuraldecoding.embedding.computeNeuronMatrixEmbeddings( ...
    X, 20, ["covXt", "corrXt"]);
features = embedding.psi.covXt.';
ranks = embedding.info.covXt.Rank;
fullRank = embedding.info.covXt.IsFullRank;
```

For these independent functions, covariance is calculated across time in double
precision; correlation is its variance-normalized counterpart. Constant neuron
traces give zero rows and columns in corrXt, including a zero diagonal, and are
recorded in `ZeroVarianceNeurons`. Pearson correlations among nonconstant neurons
are unchanged. No neurons/trials are removed and no diagonal loading is added.
Zero-rank matrices are reported as errors because the current SPSD distance
requires positive rank. Invalid/indefinite matrices also produce explicit errors.

`pairwise_manifold_distances(matrices,'SPD')` now checks every matrix, rather
than only the first. A pair uses SPD distance when both matrices are full rank;
otherwise it uses SPSD distance retaining `min(rank_i,rank_j)` dimensions. Thus
the retained dimension can differ between pairs; this is not a single fixed-rank
manifold metric. Its optional second output records `Rank`, `IsFullRank`,
`UsesSPSD`, and `PairRank`. The explicit `'SPSD'` option forces SPSD for all pairs.
Eigendecompositions are reused across pairs and principal-angle cosines are
clamped to their valid range to avoid complex distances from roundoff.

The new `computeNeuronMatrixEmbeddings` API counts **nontrivial** diffusion
coordinates: requesting 20 returns up to `min(20, trials-1)`. It saves ranks and
pairwise routing in `embedding.info`. The legacy APIs below retain their original
coordinate-count convention.

The high-level functions default to samples `[1 140]`, relative to the supplied
tensor. They do not crop or z-score data. Supply a window within the tensor;
there is no project-specific sample offset in the shared API.

The legacy diffusion parameter `maxInd` counts eigenpairs **including the trivial
one**, which is removed from `psi`. The high-level functions set it to
`min(maxDiffusionComponents, numberTrials-1)`, so the returned coordinate count
is one smaller. This convention is preserved for existing analyses. Trials must
have matching neuron identities/order and temporal coordinates. Existing geometry
and affinity behavior for degenerate inputs is retained; no new regularization or
out-of-sample embedding procedure is introduced.

## Using this from livne_slm

The original function names in `livne_slm` now forward to this package through
`setupSharedNeuralDecoding.m`. Existing calls keep their inputs and outputs.
By default the helper finds this folder beside `livne_slm`. If it is relocated,
set the `NEURAL_DATA_DECODING_ROOT` environment variable to the shared folder.
New code should use the package-qualified names above. Edit implementations
here; the project forwarding files contain no analysis logic.

The migration manifest in `migration_manifest.csv` records original classification
and regression filenames, destinations, and SHA-256 hashes.
`embedding_migration_manifest.csv` records the embedding sources and hashes before
and after migration. Both network analysis functions and the pairwise/mean-update
pipelines now explicitly call the shared embedding package; the three original
representation/embedding function names also forward here for compatibility.
Data loaders, plotting, normalization, and Python code remain in `livne_slm`.

## Verification

```matlab
setupNeuralDataDecoding;
results = runtests('tests');
assertSuccess(results);
```

The tests exercise all shared functions on synthetic data. The integration test
also checks the `livne_slm` forwarding calls when that sibling folder is present.
Embedding tests reset the path to MATLAB defaults before loading the shared
library to check that the pipeline can run independently of project folders.
