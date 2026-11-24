function [USeries, VSeries, SSeries, ...
    corrXSeries, covXSeries, ...
    corrXtSeries, covXtSeries, ...
    featureMeans, temporalContrastBasis, info, meanActivitySeries] = ...
    centered_snapshot_all_representations( ...
    X, requireFullRank, meanActivityWindow)
%CENTERED_SNAPSHOT_ALL_REPRESENTATIONS
% Compute multiple matrix representations of every snapshot.
%
% X is n-by-T-by-N:
%
%       features x time samples x snapshots
%
% For every snapshot X_k, the function computes a temporally centered
% SVD using a common temporal contrast basis:
%
%       Z_k = X_k * B
%
%       Z_k = U_k * S_k * V_k'
%
% where
%
%       B' * B = I
%       B' * ones(T,1) = 0
%
% In addition, the function computes directly from X_k:
%
%       corr(X_k)
%       cov(X_k)
%       corr(X_k')
%       cov(X_k')
%
% Inputs
% ------
% X
%   n-by-T-by-N real array.
%
% requireFullRank
%   Logical scalar. Default = true.
%   If true, an error is raised if the centered matrix Z_k does not have
%   numerical rank min(n,T-1).
%
% Outputs
% -------
% USeries
%   n-by-r-by-N array containing U_k.
%
% VSeries
%   (T-1)-by-r-by-N array containing V_k.
%
% SSeries
%   r-by-r-by-N array containing diagonal singular-value matrices S_k.
%
% corrXSeries
%   T-by-T-by-N array containing corr(X_k).
%
% covXSeries
%   T-by-T-by-N array containing cov(X_k).
%
% corrXtSeries
%   n-by-n-by-N array containing corr(X_k').
%
% covXtSeries
%   n-by-n-by-N array containing cov(X_k').
%
% featureMeans
%   n-by-N array containing temporal means of each feature.
%
% meanActivitySeries
%   n-by-1-by-N array containing the mean activity vector over the inclusive
%   sample range meanActivityWindow. In the pairwise pipeline, X has already
%   had the first 60 original samples removed and meanActivityWindow=[1 140],
%   corresponding to original trial samples 61--200.
%
% temporalContrastBasis
%   T-by-(T-1) common temporal contrast basis B.
%
% info
%   Structure containing dimensions, ranks, and rank tolerances.


if nargin < 2 || isempty(requireFullRank)
    requireFullRank = true;
end


% -------------------------------------------------------------
% Dimensions
% -------------------------------------------------------------

[n, T, numSnapshots] = size(X);

if nargin < 3 || isempty(meanActivityWindow)
    meanActivityWindow = [1, min(140, T)];
end
validateattributes(meanActivityWindow, {'numeric'}, ...
    {'vector', 'numel', 2, 'integer', 'positive', 'finite'});
meanActivityWindow = double(meanActivityWindow(:)');
if meanActivityWindow(1) > meanActivityWindow(2) || ...
        meanActivityWindow(2) > T
    error('centered_snapshot_all_representations:InvalidMeanWindow', ...
        'meanActivityWindow must be an increasing range within 1:%d.', T);
end

if T < 2
    error( ...
        'centered_snapshot_all_representations:TooFewSamples', ...
        'At least two time samples are required.' ...
    );
end

q = T - 1;
r = min(n, q);


% -------------------------------------------------------------
% Common temporal contrast basis
% -------------------------------------------------------------
%
% B spans the subspace orthogonal to the constant temporal vector.
%
% Thus X_k*B removes the temporal mean while putting all snapshots
% into identical temporal coordinates.
% -------------------------------------------------------------

temporalContrastBasis = null(ones(1, T, 'like', X));


% -------------------------------------------------------------
% Allocate outputs
% -------------------------------------------------------------

USeries = zeros( ...
    n, r, numSnapshots, ...
    'like', X ...
);

VSeries = zeros( ...
    q, r, numSnapshots, ...
    'like', X ...
);

SSeries = zeros( ...
    r, r, numSnapshots, ...
    'like', X ...
);


% corr(X) and cov(X):
%
% X is n x T.
% MATLAB treats columns as variables.
%
% Therefore these are T x T.
%
corrXSeries = zeros( ...
    T, T, numSnapshots, ...
    'like', X ...
);

covXSeries = zeros( ...
    T, T, numSnapshots, ...
    'like', X ...
);


% corr(X') and cov(X'):
%
% X' is T x n.
% Features are now the variables.
%
% Therefore these are n x n.
%
corrXtSeries = zeros( ...
    n, n, numSnapshots, ...
    'like', X ...
);

covXtSeries = zeros( ...
    n, n, numSnapshots, ...
    'like', X ...
);


featureMeans = zeros( ...
    n, numSnapshots, ...
    'like', X ...
);

meanActivitySeries = zeros( ...
    n, 1, numSnapshots, ...
    'like', X ...
);


numericalRanks = zeros( ...
    1, numSnapshots ...
);

rankTolerances = zeros( ...
    1, numSnapshots, ...
    'like', X ...
);


% -------------------------------------------------------------
% Process snapshots
% -------------------------------------------------------------

for k = 1:numSnapshots

    Xk = X(:, :, k);


    % ---------------------------------------------------------
    % Feature-wise temporal means
    % ---------------------------------------------------------

    featureMeans(:, k) = mean(Xk, 2);
    meanActivitySeries(:, 1, k) = mean( ...
        Xk(:, meanActivityWindow(1):meanActivityWindow(2)), 2);


    % ---------------------------------------------------------
    % Centered representation
    %
    % Xk*B is equivalent to projecting out the temporal mean.
    % ---------------------------------------------------------

    Zk = Xk * temporalContrastBasis;


    % ---------------------------------------------------------
    % Economy SVD
    %
    % Zk = Uk * Sk * Vk'
    % ---------------------------------------------------------

    [Uk, Sk, Vk] = svd(Zk, 'econ');

    Uk = Uk(:, 1:r);
    Vk = Vk(:, 1:r);
    Sk = Sk(1:r, 1:r);

    sk = diag(Sk);


    % ---------------------------------------------------------
    % Numerical rank
    % ---------------------------------------------------------

    if isempty(sk) || sk(1) == 0

        rankTolerance = zeros(1, 'like', X);

    else

        rankTolerance = ...
            max(n, q) * eps(class(X)) * sk(1);

    end

    numericalRank = sum(sk > rankTolerance);

    numericalRanks(k) = numericalRank;
    rankTolerances(k) = rankTolerance;


    if requireFullRank && numericalRank < r

        error( ...
            'centered_snapshot_all_representations:RankDeficient', ...
            ['Centered snapshot %d has numerical rank %d; ' ...
             'expected %d.'], ...
            k, numericalRank, r ...
        );

    end


    % ---------------------------------------------------------
    % Store both SVD factors
    %
    % No decision between U and V is made.
    % ---------------------------------------------------------
    [USeries(:, :, k), ~, VSeries(:, :, k)] = orient_svd_so(Uk, Sk, Vk);
 
    SSeries(:, :, k) = Sk;


    % ---------------------------------------------------------
    % Correlation and covariance across columns of X
    %
    % Xk is n x T.
    %
    % Observations = features
    % Variables    = time samples
    %
    % Result: T x T
    % ---------------------------------------------------------

    corrX = corr(Xk);
    covX = cov(Xk);


    % ---------------------------------------------------------
    % Correlation and covariance across columns of X'
    %
    % Xk' is T x n.
    %
    % Observations = time samples
    % Variables    = features
    %
    % Result: n x n
    % ---------------------------------------------------------

    corrXt = corr(Xk');
    covXt = cov(Xk');


    % ---------------------------------------------------------
    % Enforce numerical symmetry
    % ---------------------------------------------------------

    corrX = 0.5 * (corrX + corrX');
    covX = 0.5 * (covX + covX');

    corrXt = 0.5 * (corrXt + corrXt');
    covXt = 0.5 * (covXt + covXt');


    % ---------------------------------------------------------
    % Store
    % ---------------------------------------------------------

    corrXSeries(:, :, k) = corrX;
    covXSeries(:, :, k) = covX;

    corrXtSeries(:, :, k) = corrXt;
    covXtSeries(:, :, k) = covXt;

end


% -------------------------------------------------------------
% Information structure
% -------------------------------------------------------------

info = struct();

info.numFeatures = n;

info.numOriginalTimeSamples = T;

info.numCenteredTimeCoordinates = q;

info.numSnapshots = numSnapshots;

info.rank = r;

info.numericalRank = numericalRanks;

info.rankTolerance = rankTolerances;

info.isFullRank = numericalRanks == r;

info.requireFullRank = requireFullRank;

info.Udimensions = [n, r];

info.Vdimensions = [q, r];

info.Sdimensions = [r, r];

info.corrXdimensions = [T, T];

info.covXdimensions = [T, T];

info.corrXtdimensions = [n, n];

info.covXtdimensions = [n, n];

info.meanActivityDimensions = [n, 1];

info.meanActivityWindow = meanActivityWindow;

info.meanActivityNumberSamples = diff(meanActivityWindow) + 1;

end
