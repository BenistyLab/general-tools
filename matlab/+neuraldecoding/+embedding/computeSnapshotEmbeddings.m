function embedding = computeSnapshotEmbeddings( ...
    X, maxDiffusionComponents, meanActivityWindow, representationNames)
%COMPUTESNAPSHOTEMBEDDINGS Build the eight snapshot diffusion embeddings.
% Optional fourth input selects one or more representation names:
% soU, soV, sv, corrX, covX, corrXt, covXt, meanActivity.
% Default: all eight, in the original order. Only selected distance matrices
% and embeddings are computed. covXt/corrXt/meanActivity can be selected
% independently, without a centered full-rank SVD requirement. Selecting
% soU/soV/sv/corrX/covX retains the legacy shared-SVD requirement.
% psi.(name) is components-by-trials; transpose it for decoder features.

if nargin < 2 || isempty(maxDiffusionComponents)
    maxDiffusionComponents = 20;
end
if nargin < 3 || isempty(meanActivityWindow)
    meanActivityWindow = [1 140];
end

allNames = {'soU', 'soV', 'sv', 'corrX', 'covX', ...
    'corrXt', 'covXt', 'meanActivity'};
if nargin < 4
    representationNames = allNames;
end
names = validateRepresentationNames(representationNames, allNames);
validateattributes(X, {'single', 'double'}, ...
    {'real', 'finite', 'nonempty'}, mfilename, 'X');
if ndims(X) > 3 || size(X, 1) < 2 || size(X, 2) < 2 || size(X, 3) < 2
    error('computeSnapshotEmbeddings:InvalidDimensions', ...
        'X must be neurons-by-time-by-trials with at least two in each dimension.');
end
validateattributes(maxDiffusionComponents, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'}, mfilename, ...
    'maxDiffusionComponents');

legacyNames = {'soU','soV','sv','corrX','covX'};
series = cell(1,numel(allNames));
featureMeans = reshape(mean(X,2),size(X,1),size(X,3));
temporalContrastBasis = [];
info = struct;
if any(ismember(names,legacyNames))
    [USeries, VSeries, SSeries, ...
    corrXSeries, covXSeries, ...
    ~, ~, ...
    featureMeans, temporalContrastBasis, info, meanActivitySeries] = ...
    neuraldecoding.embedding.centered_snapshot_all_representations( ...
        X, true, meanActivityWindow);
    series(1:5) = {USeries,VSeries,SSeries,corrXSeries,covXSeries};
else
    validateattributes(meanActivityWindow,{'numeric'}, ...
        {'vector','numel',2,'integer','positive','finite'});
    assert(meanActivityWindow(1)<=meanActivityWindow(2) && ...
        meanActivityWindow(2)<=size(X,2),'computeSnapshotEmbeddings:MeanWindow', ...
        'Mean window must be an inclusive range inside the trial.');
    meanActivitySeries = mean(X(:,meanActivityWindow(1):meanActivityWindow(2),:),2);
end
neuronNames = names(ismember(names,{'corrXt','covXt'}));
if ~isempty(neuronNames)
    [neuronMatrices,neuronInfo] = neuraldecoding.embedding.extractNeuronMatrices(X,neuronNames);
    info.neuronMatrices = neuronInfo;
    for i=1:numel(neuronNames)
        index=strcmp(neuronNames{i},allNames);
        series{index}=neuronMatrices.(neuronNames{i});
    end
end

distance = struct;
series{8} = meanActivitySeries;
geometries = {'SO', 'SO', 'EUC', 'SPD', 'SPD', 'SPD', 'SPD', 'EUC'};
for nameIndex = 1:numel(names)
    index = find(strcmp(names{nameIndex}, allNames));
    [distance.(names{nameIndex}),distanceInfo] = ...
        neuraldecoding.embedding.pairwise_manifold_distances( ...
            series{index}, geometries{index});
    if ismember(names{nameIndex},{'corrXt','covXt'})
        info.neuronMatrices.(names{nameIndex}).Distance = distanceInfo;
    end
end
dParams.maxInd = min(maxDiffusionComponents, size(X, 3) - 1);
affinity = struct;
psi = struct;
for nameIndex = 1:numel(names)
    name = names{nameIndex};
    affinity.(name) = neuraldecoding.embedding.dist2affinity(distance.(name));
    psi.(name) = neuraldecoding.embedding.calcDiffusionMap(affinity.(name), dParams);
end

embedding = struct;
embedding.distance = distance;
embedding.affinity = affinity;
embedding.psi = psi;
embedding.representationNames = names;
embedding.featureMeans = featureMeans;
embedding.meanActivity = meanActivitySeries;
embedding.temporalContrastBasis = temporalContrastBasis;
embedding.info = info;
embedding.diffusionParameters = dParams;
end

function names = validateRepresentationNames(requested, allNames)
if ~(ischar(requested) || isstring(requested) || iscellstr(requested))
    error('computeSnapshotEmbeddings:InvalidRepresentations', ...
        'Representation names must be text.');
end
requested = string(requested);
if ~isvector(requested) || isempty(requested) || any(ismissing(requested)) || ...
        any(~ismember(requested, string(allNames))) || ...
        numel(unique(requested)) ~= numel(requested)
    error('computeSnapshotEmbeddings:InvalidRepresentations', ...
        'Choose distinct names from: %s.', strjoin(allNames, ', '));
end
names = cellstr(requested(:).');
end
