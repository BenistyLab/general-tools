function embedding = computeMeanActivityEmbedding( ...
    X, maxDiffusionComponents, meanActivityWindow)
%COMPUTEMEANACTIVITYEMBEDDING Embed only trial-mean neural activity.
%
% X is cells-by-time-by-trials. The caller is responsible for any desired
% activity normalization. No other snapshot representation is calculated.

if nargin < 2 || isempty(maxDiffusionComponents)
    maxDiffusionComponents = 20;
end
if nargin < 3 || isempty(meanActivityWindow)
    meanActivityWindow = [1 140];
end

validateattributes(X, {'numeric'}, {'real', 'nonempty', '3d'}, ...
    mfilename, 'X');
validateattributes(maxDiffusionComponents, {'numeric'}, ...
    {'scalar', 'integer', 'positive'}, mfilename, ...
    'maxDiffusionComponents');
validateattributes(meanActivityWindow, {'numeric'}, ...
    {'vector', 'numel', 2, 'integer', 'positive', 'increasing'}, ...
    mfilename, 'meanActivityWindow');

meanActivityWindow = double(meanActivityWindow(:)');
if meanActivityWindow(2) > size(X, 2)
    error('computeMeanActivityEmbedding:WindowOutsideTrial', ...
        ['Mean-activity window [%d %d] exceeds the retained trial ' ...
         'duration of %d samples.'], meanActivityWindow(1), ...
        meanActivityWindow(2), size(X, 2));
end
if size(X, 3) < 2
    error('computeMeanActivityEmbedding:TooFewTrials', ...
        'At least two trials are required for a diffusion embedding.');
end

meanActivity = mean( ...
    X(:, meanActivityWindow(1):meanActivityWindow(2), :), 2);
distance = neuraldecoding.embedding.pairwise_manifold_distances(meanActivity, 'EUC');
affinity = neuraldecoding.embedding.dist2affinity(distance);
diffusionParameters.maxInd = min( ...
    maxDiffusionComponents, size(X, 3) - 1);
psi = neuraldecoding.embedding.calcDiffusionMap(affinity, diffusionParameters);

embedding = struct;
embedding.distance = distance;
embedding.affinity = affinity;
embedding.psi = psi;
embedding.meanActivity = meanActivity;
embedding.meanActivityWindow = meanActivityWindow;
embedding.representationName = 'meanActivity';
embedding.diffusionParameters = diffusionParameters;
end
